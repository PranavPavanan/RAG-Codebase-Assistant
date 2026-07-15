"""Repository indexer service for processing and indexing repository content."""
import asyncio
import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import time
import os
import stat

from src.models.index import (
    FileIndexEntry,
    IndexingStatus,
    IndexProgressInfo,
    IndexStartRequest,
    IndexStartResponse,
    IndexStats,
    IndexStatusResponse,
)


class IndexerService:
    """Service for indexing repository content."""

    def __init__(self, storage_dir: str = "./storage"):
        """
        Initialize indexer service.

        Args:
            storage_dir: Directory for storing indexed data and cloned repos
        """
        self.storage_dir = Path(storage_dir)
        self.repos_dir = self.storage_dir / "repositories"
        self.indices_dir = self.storage_dir / "indices"
        self.metadata_dir = self.storage_dir / "metadata"

        # Create directories if they don't exist
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # In-memory task tracking
        self.active_tasks = {}
        self.current_task = None

    def start_indexing(self, request: IndexStartRequest) -> IndexStartResponse:
        """
        Start indexing a repository.

        Args:
            request: Index start request with repository URL

        Returns:
            IndexStartResponse with task ID
        """
        # Check if repository is already indexed
        existing_repo = self._check_existing_repository(request.repository_url)
        if existing_repo:
            return IndexStartResponse(
                task_id="already-indexed",
                status=IndexingStatus.COMPLETED,
                message=f"Repository '{existing_repo['repository_name']}' is already available in the database. Last indexed: {existing_repo['indexed_at']}",
                repository_url=request.repository_url,
                estimated_time=0,
                created_at=existing_repo['created_at'],
            )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Create task record
        task = {
            "task_id": task_id,
            "status": IndexingStatus.PENDING,
            "repository_url": request.repository_url,
            "branch": request.branch,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "progress": IndexProgressInfo(
                files_processed=0,
                total_files=0,
                percentage=0.0,
            ),
            "error": None,
            "result": None,
        }

        self.active_tasks[task_id] = task
        self.current_task = task_id

        # Start background indexing task
        asyncio.create_task(self._index_repository_async(task_id, request))

        return IndexStartResponse(
            task_id=task_id,
            status=IndexingStatus.PENDING,
            message="Indexing task created and queued",
            repository_url=request.repository_url,
            estimated_time=300,  # 5 minutes estimate
            created_at=task["created_at"],
        )

    def get_indexing_status(self, task_id: str) -> Optional[IndexStatusResponse]:
        """
        Get indexing task status.

        Args:
            task_id: Task identifier

        Returns:
            IndexStatusResponse or None if task not found
        """
        task = self.active_tasks.get(task_id)
        if not task:
            return None

        return IndexStatusResponse(
            task_id=task_id,
            status=task["status"],
            message=self._get_status_message(task["status"]),
            progress=task["progress"],
            percentage=task["progress"].percentage,
            repository_url=task["repository_url"],
            started_at=task["started_at"],
            completed_at=task["completed_at"],
            error=task["error"],
            result=task["result"],
        )

    def get_task_status(self, task_id: str) -> Optional[IndexStatusResponse]:
        """
        Get task status (alias for get_indexing_status).

        Args:
            task_id: Task identifier

        Returns:
            IndexStatusResponse or None if task not found
        """
        return self.get_indexing_status(task_id)

    def clear_index(self) -> dict:
        """
        Clear current index.

        Returns:
            Dictionary with operation result
        """
        files_removed = 0
        space_freed = 0

        try:
            # Remove index files
            if self.indices_dir.exists():
                for file in self.indices_dir.glob("*"):
                    if file.is_file():
                        space_freed += file.stat().st_size
                        file.unlink()
                        files_removed += 1

            # Clear metadata
            if self.metadata_dir.exists():
                for file in self.metadata_dir.glob("*"):
                    if file.is_file():
                        file.unlink()

            return {
                "success": True,
                "files_removed": files_removed,
                "space_freed": space_freed,
                "message": "Index cleared successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to clear index",
            }

    def get_index_stats(self) -> IndexStats:
        """
        Get current index statistics.

        Returns:
            IndexStats with current index information
        """
        # Check if any index exists - look for both FAISS files and metadata files
        index_files = list(self.indices_dir.glob("*.faiss")) if self.indices_dir.exists() else []
        metadata_files = (
            list(self.metadata_dir.glob("*.json")) if self.metadata_dir.exists() else []
        )

        # If no metadata files exist, there's no indexed repository
        if not metadata_files:
            return IndexStats(
                is_indexed=False,
                repository_name=None,
                file_count=0,
                total_size=0,
                vector_count=0,
                last_updated=None,
                created_at=None,
            )

        # Try to load metadata from the most recent metadata file
        repository_name = None
        last_updated = None
        created_at = None
        file_count = 0
        total_size = 0

        # Sort metadata files by modification time and get the most recent
        if metadata_files:
            try:
                latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
                import json

                with open(latest_metadata) as f:
                    metadata = json.load(f)
                    repository_name = metadata.get("repository_name")
                    file_count = metadata.get("file_count", 0)
                    
                    # Calculate total size from file entries in metadata
                    files = metadata.get("files", [])
                    total_size = sum(file_info.get("size", 0) for file_info in files)
                    
                    if metadata.get("indexed_at"):
                        last_updated = datetime.fromisoformat(metadata["indexed_at"])
                    if metadata.get("created_at"):
                        created_at = datetime.fromisoformat(metadata["created_at"])
            except Exception as e:
                print(f"Error loading metadata: {e}")
                # If metadata loading fails, still consider it indexed if files exist
                if metadata_files:
                    return IndexStats(
                        is_indexed=True,
                        repository_name="Unknown Repository",
                        file_count=0,
                        total_size=0,
                        vector_count=0,
                        last_updated=None,
                        created_at=None,
                    )

        # Also check for FAISS files for additional size calculation
        if index_files:
            faiss_size = sum(f.stat().st_size for f in index_files if f.is_file())
            total_size += faiss_size

        return IndexStats(
            is_indexed=True,
            repository_name=repository_name,
            file_count=file_count,
            total_size=total_size,
            vector_count=file_count * 10,  # Estimate: ~10 vectors per file
            last_updated=last_updated,
            created_at=created_at,
        )

    async def _index_repository_async(self, task_id: str, request: IndexStartRequest):
        """
        Asynchronously index a repository with real progress tracking.
        
        Args:
            task_id: Task identifier
            request: Index start request
        """
        task = self.active_tasks.get(task_id)
        if not task:
            return

        try:
            # Update status to running
            task["status"] = IndexingStatus.RUNNING
            task["started_at"] = datetime.utcnow()
            start_time = time.time()

            # Step 1: Clone repository
            await self._update_progress(task_id, "Cloning repository...", 0, 0, 0)
            repo_path = await self._clone_repository(request.repository_url, request.branch)
            
            # Step 2: Discover files
            await self._update_progress(task_id, "Discovering files...", 0, 0, 5)
            files_to_process = self._discover_files(repo_path, [], [])
            
            # Update total files count
            task["progress"].total_files = len(files_to_process)
            await self._update_progress(task_id, f"Found {len(files_to_process)} files to process", 0, len(files_to_process), 10)

            # Step 3: Process files with deduplication
            processed_files = []
            processed_file_paths = set()  # Track processed file paths for deduplication
            total_bytes = 0
            
            for i, file_path in enumerate(files_to_process):
                # Update current file being processed
                relative_path = str(file_path.relative_to(repo_path))
                
                # Skip if file already processed (deduplication)
                if relative_path in processed_file_paths:
                    continue
                processed_file_paths.add(relative_path)
                
                # Process file
                file_entry = self.index_file(file_path, repo_path)
                if file_entry:
                    processed_files.append(file_entry)
                    total_bytes += file_entry.size
                
                # Update progress
                progress_percentage = 10 + (i + 1) / len(files_to_process) * 80  # 10-90% for file processing
                task["progress"].files_processed = i + 1
                task["progress"].percentage = progress_percentage
                
                # Calculate elapsed time and estimated remaining for internal tracking
                elapsed = time.time() - start_time
                
                await self._update_progress(
                    task_id, 
                    f"Processing {relative_path} ({i + 1}/{len(files_to_process)})", 
                    i + 1, 
                    len(files_to_process), 
                    progress_percentage
                )
                
                # Small delay to show progress
                await asyncio.sleep(0.1)

            # Step 4: Create index metadata
            await self._update_progress(task_id, "Creating index metadata...", len(files_to_process), len(files_to_process), 90)
            await self._save_index_metadata(task_id, processed_files, request.repository_url)

            # Step 5: Complete
            task["status"] = IndexingStatus.COMPLETED
            task["completed_at"] = datetime.utcnow()
            task["result"] = {
                "files_processed": len(processed_files),
                "total_bytes": total_bytes,
                "processing_time": time.time() - start_time,
            }
            
            await self._update_progress(task_id, "Indexing completed successfully!", len(files_to_process), len(files_to_process), 100)

        except Exception as e:
            task["status"] = IndexingStatus.FAILED
            task["error"] = str(e)
            task["completed_at"] = datetime.utcnow()
            await self._update_progress(task_id, f"Indexing failed: {str(e)}", 0, 0, 0)

    async def _update_progress(self, task_id: str, message: str, files_processed: int, total_files: int, percentage: float):
        """Update task progress and send WebSocket notification."""
        task = self.active_tasks.get(task_id)
        if task:
            task["message"] = message
            task["progress"].files_processed = files_processed
            task["progress"].total_files = total_files
            task["progress"].percentage = percentage

    async def _clone_repository(self, repo_url: str, branch: Optional[str]) -> Path:
        """Clone repository to local storage."""
        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = self.repos_dir / repo_name
        
        # Remove existing directory if it exists
        if repo_path.exists():
            def remove_readonly(func, path, _):
                # Set the file to writable and retry the operation
                os.chmod(path, stat.S_IWRITE)
                func(path)
                
            shutil.rmtree(repo_path, onerror=remove_readonly)
        
        # Clone repository
        try:
            clone_cmd = [
                "git", "clone", 
                "--depth", "1"
            ]
            
            # Let git natively fetch the repository's default branch if none is provided.
            if branch:
                clone_cmd.extend(["--branch", branch])
                
            clone_cmd.extend([repo_url, str(repo_path)])
            
            subprocess.run(clone_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to clone repository: {e.stderr.decode()}")
        
        return repo_path

    def _discover_files(self, repo_path: Path, include_patterns: List[str], exclude_patterns: List[str]) -> List[Path]:
        """Discover files to process based on include/exclude patterns."""
        files = []
        
        # Default include patterns if none provided
        if not include_patterns:
            include_patterns = ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml"]
        
        # Default exclude patterns
        default_exclude = [".git/**", "__pycache__/**", "*.pyc", "venv/**", "node_modules/**", "*.log"]
        if exclude_patterns:
            exclude_patterns.extend(default_exclude)
        else:
            exclude_patterns = default_exclude
        
        # Find files matching include patterns
        for pattern in include_patterns:
            files.extend(repo_path.rglob(pattern))
        
        # Filter out excluded files
        filtered_files = []
        for file_path in files:
            if file_path.is_file():
                relative_path = str(file_path.relative_to(repo_path))
                should_exclude = False
                
                for exclude_pattern in exclude_patterns:
                    if self._matches_pattern(relative_path, exclude_pattern):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    filtered_files.append(file_path)
        
        return filtered_files

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a glob pattern."""
        from fnmatch import fnmatch
        return fnmatch(file_path, pattern)

    async def _save_index_metadata(self, task_id: str, processed_files: List[FileIndexEntry], repo_url: str):
        """Save index metadata to file with deduplication."""
        # Deduplicate files by file_path to prevent duplicate entries
        unique_files = {}
        for file_entry in processed_files:
            file_path = file_entry.file_path
            if file_path not in unique_files:
                unique_files[file_path] = file_entry
            else:
                # Keep the most recent entry if duplicates found
                if file_entry.indexed_at > unique_files[file_path].indexed_at:
                    unique_files[file_path] = file_entry
        
        # Convert back to list
        deduplicated_files = list(unique_files.values())
        
        metadata = {
            "task_id": task_id,
            "repository_url": repo_url,
            "repository_name": repo_url.split("/")[-1].replace(".git", ""),
            "file_count": len(deduplicated_files),
            "files": [file_entry.dict() for file_entry in deduplicated_files],
            "indexed_at": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        metadata_file = self.metadata_dir / f"{task_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def index_file(self, file_path: Path, repo_root: Path) -> Optional[FileIndexEntry]:
        """
        Index a single file.

        Args:
            file_path: Path to file to index
            repo_root: Root path of repository

        Returns:
            FileIndexEntry or None if file should be skipped
        """
        try:
            # Skip binary files and large files
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                return None

            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return None  # Skip binary files

            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Get relative path
            relative_path = str(file_path.relative_to(repo_root))

            # Detect language from extension
            language = self._detect_language(file_path.suffix)

            return FileIndexEntry(
                file_path=relative_path,
                content_hash=content_hash,
                size=len(content.encode()),
                language=language,
                chunk_count=len(content) // 512 + 1,  # Rough chunk estimate
                indexed_at=datetime.utcnow(),
            )

        except Exception:
            return None

    def _get_status_message(self, status: IndexingStatus) -> str:
        """Get human-readable status message."""
        messages = {
            IndexingStatus.PENDING: "Task is queued and waiting to start",
            IndexingStatus.RUNNING: "Indexing in progress",
            IndexingStatus.COMPLETED: "Indexing completed successfully",
            IndexingStatus.FAILED: "Indexing failed",
            IndexingStatus.CANCELLED: "Indexing was cancelled",
        }
        return messages.get(status, "Unknown status")

    def _check_existing_repository(self, repository_url: str) -> Optional[dict]:
        """
        Check if a repository is already indexed.
        
        Args:
            repository_url: Repository URL to check
            
        Returns:
            Repository metadata if found, None otherwise
        """
        try:
            if not self.metadata_dir.exists():
                return None
            
            # Get all metadata files
            metadata_files = list(self.metadata_dir.glob("*.json"))
            if not metadata_files:
                return None
            
            # Check each metadata file for matching repository URL
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('repository_url') == repository_url:
                        return {
                            'repository_name': metadata.get('repository_name'),
                            'indexed_at': metadata.get('indexed_at'),
                            'created_at': metadata.get('created_at'),
                            'file_count': metadata.get('file_count', 0)
                        }
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error checking existing repository: {e}")
            return None

    def _detect_language(self, extension: str) -> Optional[str]:
        """Detect programming language from file extension."""
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "shell",
            ".bash": "shell",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
        }
        return lang_map.get(extension.lower())


# Singleton instance
_indexer_service: Optional[IndexerService] = None


def get_indexer_service(storage_dir: str = "./storage") -> IndexerService:
    """
    Get or create indexer service instance.

    Args:
        storage_dir: Storage directory path

    Returns:
        IndexerService instance
    """
    global _indexer_service
    if _indexer_service is None:
        _indexer_service = IndexerService(storage_dir)
    return _indexer_service
