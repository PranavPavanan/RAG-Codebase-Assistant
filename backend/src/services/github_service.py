"""GitHub API service for repository operations."""
import re
from typing import List, Optional

from github import Github, GithubException
from github import Repository as GHRepository

from src.config import settings
from src.models.repository import (
    Repository,
    RepositoryBase,
    RepositorySearchRequest,
    RepositorySearchResponse,
    RepositoryTrendingRequest,
    RepositoryValidationRequest,
    RepositoryValidationResponse,
)


class GitHubService:
    """Service for interacting with GitHub API."""

    def __init__(self, github_token: Optional[str] = None):
        self.github = Github(github_token) if github_token else Github()
        self.github_token = github_token

    # ── Mode 1: Find a specific repo by name/user ──────────────────────────
    def search_repositories(self, request: RepositorySearchRequest) -> RepositorySearchResponse:
        """
        Search for repositories by repo name and/or username.

        Strategy:
          - If a GitHub URL is given, fetch the exact repo and pin it first.
          - If username is provided, build `user:<username> <name> in:name`.
          - Otherwise search `<name> in:name` with best-match ranking so
            exact-name matches always outrank star count.
        """
        try:
            repo_name = (request.query or "").strip()
            username  = (request.username or "").strip()

            # Direct fetch if user gave a full URL
            direct_repo = None
            github_url_re = r"^https?://github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$"
            url_match = re.match(github_url_re, repo_name)
            if url_match:
                owner, rname = url_match.groups()
                try:
                    direct_repo = self.github.get_repo(f"{owner}/{rname}")
                except Exception:
                    pass
                repo_name = rname
                if not username:
                    username = owner

            # Direct fetch if user gave owner/repo shorthand
            owner_repo_re = r"^([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)$"
            or_match = re.match(owner_repo_re, repo_name)
            if or_match and not direct_repo:
                owner, rname = or_match.groups()
                try:
                    direct_repo = self.github.get_repo(f"{owner}/{rname}")
                except Exception:
                    pass
                if not username:
                    username = owner
                repo_name = rname

            # Build the GitHub search qualifier string
            parts = []
            if repo_name:
                parts.append(f"{repo_name} in:name")
            if username:
                parts.append(f"user:{username}")
            if not parts:
                raise ValueError("Please enter a repository name or username.")

            search_query = " ".join(parts)

            # By omitting sort/order, GitHub defaults to best-match (relevance)
            repositories = self.github.search_repositories(query=search_query)

            results: List[RepositoryBase] = []

            # Pin directly-fetched repo at the top
            if direct_repo:
                results.append(self._convert_to_repository_base(direct_repo))

            for repo in repositories:
                if len(results) >= request.limit:
                    break
                if direct_repo and repo.id == direct_repo.id:
                    continue
                results.append(self._convert_to_repository_base(repo))

            return RepositorySearchResponse(
                repositories=results,
                total_count=repositories.totalCount,
                page=1,
            )

        except GithubException as e:
            raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}") from e

    # ── Mode 2: Trending repos sorted by stars ─────────────────────────────
    def search_trending_repositories(self, request: RepositoryTrendingRequest) -> RepositorySearchResponse:
        """
        Fetch top repositories sorted by star count.

        Filters available (all optional):
          - keyword  : any text to include in the search (e.g. 'machine learning')
          - language : programming language filter (e.g. 'python', 'typescript')
          - topic    : GitHub topic tag (e.g. 'web', 'cli', 'api')
          - min_stars: only include repos with at least this many stars (default 1000)

        Returns repos sorted by stars descending.
        """
        try:
            parts = []
            if request.keyword:
                parts.append(request.keyword.strip())
            if request.language:
                parts.append(f"language:{request.language.strip()}")
            if request.topic:
                parts.append(f"topic:{request.topic.strip()}")

            min_stars = request.min_stars if request.min_stars is not None else 1000
            parts.append(f"stars:>={min_stars}")

            # Always exclude archived repos for quality results
            parts.append("archived:false")

            search_query = " ".join(parts) if parts else "stars:>=10000"

            repositories = self.github.search_repositories(
                query=search_query, sort="stars", order="desc"
            )

            results: List[RepositoryBase] = []
            for repo in repositories:
                if len(results) >= request.limit:
                    break
                results.append(self._convert_to_repository_base(repo))

            return RepositorySearchResponse(
                repositories=results,
                total_count=repositories.totalCount,
                page=1,
            )

        except GithubException as e:
            raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
        except Exception as e:
            raise ValueError(f"Trending search failed: {str(e)}") from e

    # ── Validation & helpers ───────────────────────────────────────────────
    def validate_repository_url(self, request: RepositoryValidationRequest) -> RepositoryValidationResponse:
        """Validate a GitHub repository URL."""
        url = request.url.strip()
        github_pattern = r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
        match = re.match(github_pattern, url)

        if not match:
            return RepositoryValidationResponse(
                valid=False,
                message="Invalid GitHub repository URL format. Expected: https://github.com/owner/repo",
            )

        owner, repo_name = match.groups()
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            repo_base = self._convert_to_repository_base(repo)
            return RepositoryValidationResponse(
                valid=True,
                message="Repository is valid and accessible",
                repository_info=repo_base,
            )
        except GithubException as e:
            if e.status == 404:
                return RepositoryValidationResponse(valid=False, message="Repository not found or not accessible")
            return RepositoryValidationResponse(valid=False, message=f"Error accessing repository: {e.data.get('message', str(e))}")
        except Exception as e:
            return RepositoryValidationResponse(valid=False, message=f"Validation error: {str(e)}")

    def get_repository(self, repo_id: str) -> Optional[Repository]:
        try:
            repo = self.github.get_repo(repo_id)
            return self._convert_to_repository(repo)
        except Exception:
            return None

    def get_repository_by_url(self, url: str) -> Optional[Repository]:
        github_pattern = r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
        match = re.match(github_pattern, url)
        if not match:
            return None
        owner, repo_name = match.groups()
        return self.get_repository(f"{owner}/{repo_name}")

    def clone_repository(self, repo_url: str, target_dir: str) -> bool:
        try:
            import git
            git.Repo.clone_from(repo_url, target_dir)
            return True
        except Exception as e:
            print(f"Clone failed: {e}")
            return False

    def _convert_to_repository_base(self, gh_repo: GHRepository) -> RepositoryBase:
        """Convert a PyGithub Repository object to our RepositoryBase model."""
        return RepositoryBase(
            id=str(gh_repo.id),
            name=gh_repo.name,
            full_name=gh_repo.full_name,
            description=gh_repo.description,
            url=gh_repo.html_url,
            html_url=gh_repo.html_url,
            stars=gh_repo.stargazers_count,
            stargazers_count=gh_repo.stargazers_count,
            forks=gh_repo.forks_count,
            language=gh_repo.language,
            topics=gh_repo.get_topics() if hasattr(gh_repo, "get_topics") else [],
            owner=gh_repo.owner.login,
            default_branch=gh_repo.default_branch,
            size=gh_repo.size,
            updated_at=gh_repo.updated_at,
            created_at=gh_repo.created_at,
        )

    def _convert_to_repository(self, gh_repo: GHRepository) -> Repository:
        """Convert a PyGithub Repository object to our full Repository model."""
        base_dict = self._convert_to_repository_base(gh_repo).model_dump()
        return Repository(
            **base_dict,
            clone_url=gh_repo.clone_url,
            ssh_url=gh_repo.ssh_url,
            open_issues=gh_repo.open_issues_count,
            watchers=gh_repo.watchers_count,
            license=gh_repo.license.name if gh_repo.license else None,
            is_private=gh_repo.private,
            is_fork=gh_repo.fork,
            has_wiki=gh_repo.has_wiki,
            has_issues=gh_repo.has_issues,
        )


# Singleton instance
_github_service: Optional[GitHubService] = None


def get_github_service(github_token: Optional[str] = None) -> GitHubService:
    global _github_service
    if _github_service is None:
        token = github_token or settings.get_github_token()
        _github_service = GitHubService(token)
    return _github_service
