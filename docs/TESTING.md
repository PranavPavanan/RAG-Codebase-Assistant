# Testing Guide

This document provides comprehensive information about the testing setup for the RAG GitHub Assistant project.

## Overview

The project includes a complete testing suite covering:
- **Backend**: Unit tests, integration tests, and end-to-end tests
- **Frontend**: Component tests and integration tests
- **WebSocket**: Real-time communication tests
- **API**: All 12 REST endpoints tested

## Test Structure

```
backend/
├── tests/
│   ├── unit/                    # Unit tests for services and models
│   │   ├── test_github_service.py
│   │   ├── test_indexer_service.py
│   │   ├── test_rag_service.py
│   │   ├── test_models.py
│   │   ├── test_api_health.py
│   │   ├── test_api_search.py
│   │   ├── test_api_repositories.py
│   │   ├── test_api_indexing.py
│   │   ├── test_api_chat.py
│   │   └── test_websocket.py
│   ├── integration/             # Integration tests for workflows
│   │   ├── test_repository_workflow.py
│   │   └── test_websocket_workflow.py
│   └── e2e/                     # End-to-end tests
│       └── test_complete_workflow.py
├── run_tests.py                 # Test runner script
└── pytest.ini                  # Pytest configuration

frontend/
├── tests/
│   ├── unit/                    # Component unit tests
│   │   └── test_components.test.tsx
│   └── integration/             # User workflow tests
│       └── test_user_workflows.test.tsx
├── run_tests.js                 # Test runner script
└── vitest.config.ts            # Vitest configuration
```

## Running Tests

### Backend Tests

#### Run All Tests
```bash
cd backend
python run_tests.py
```

#### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# E2E tests only
python -m pytest tests/e2e/ -v

# Specific test file
python -m pytest tests/unit/test_github_service.py -v
```

#### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Frontend Tests

#### Run All Tests
```bash
cd frontend
node run_tests.js
```

#### Run Specific Test Categories
```bash
# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# All tests
npm run test

# With coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

## Test Categories

### Backend Tests

#### Unit Tests
- **GitHub Service** (`test_github_service.py`): Tests repository search, validation, and retrieval
- **Indexer Service** (`test_indexer_service.py`): Tests indexing operations and progress tracking
- **RAG Service** (`test_rag_service.py`): Tests chat functionality and model interactions
- **Models** (`test_models.py`): Tests data model validation and serialization
- **API Endpoints** (`test_api_*.py`): Tests all 12 REST API endpoints
- **WebSocket** (`test_websocket.py`): Tests real-time communication

#### Integration Tests
- **Repository Workflow** (`test_repository_workflow.py`): Complete workflow from search to indexing
- **WebSocket Workflow** (`test_websocket_workflow.py`): Real-time updates and messaging

#### End-to-End Tests
- **Complete Workflow** (`test_complete_workflow.py`): Full user journey from search to chat

### Frontend Tests

#### Unit Tests
- **Component Tests** (`test_components.test.tsx`): Individual React component testing
- **Store Tests**: Zustand state management testing

#### Integration Tests
- **User Workflows** (`test_user_workflows.test.tsx`): Complete user interaction flows

## Test Features

### Backend Testing Features
- **Mocking**: Comprehensive mocking of external services (GitHub API, RAG model)
- **Fixtures**: Reusable test data and setup
- **Error Handling**: Tests for various error conditions
- **Performance**: Tests for concurrent operations and large data handling
- **Validation**: Input validation and error response testing
- **WebSocket**: Real-time communication testing

### Frontend Testing Features
- **Component Testing**: Individual component behavior and rendering
- **User Interactions**: Click, input, keyboard navigation testing
- **State Management**: Zustand store state changes
- **API Integration**: Mock API calls and responses
- **Error Handling**: Error state display and recovery
- **Accessibility**: Basic accessibility testing

## Test Data

### Mock Data
- **Repository Data**: Sample GitHub repositories with various properties
- **Indexing Data**: Mock indexing progress and statistics
- **Chat Data**: Sample conversations and responses
- **Error Data**: Various error conditions and responses

### Test Scenarios
- **Happy Path**: Normal user workflows
- **Error Cases**: Network errors, validation failures, service errors
- **Edge Cases**: Empty inputs, large data, concurrent operations
- **Performance**: Multiple users, large repositories, long conversations

## Configuration

### Backend Configuration
- **pytest.ini**: Pytest configuration with markers and options
- **Coverage**: HTML and terminal coverage reports
- **Markers**: Unit, integration, e2e, slow, websocket, api, services, models

### Frontend Configuration
- **vitest.config.ts**: Vitest configuration
- **Test Environment**: jsdom for DOM testing
- **Mocking**: API and WebSocket client mocking

## Continuous Integration

### GitHub Actions (Recommended)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
      - name: Run tests
        run: |
          cd backend
          python run_tests.py

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      - name: Run tests
        run: |
          cd frontend
          node run_tests.js
```

## Best Practices

### Writing Tests
1. **Arrange-Act-Assert**: Structure tests clearly
2. **Descriptive Names**: Use clear, descriptive test names
3. **Single Responsibility**: One test per behavior
4. **Mock External Dependencies**: Don't rely on external services
5. **Test Edge Cases**: Include error conditions and boundary cases
6. **Clean Up**: Properly clean up after tests

### Test Organization
1. **Group Related Tests**: Use describe blocks for related functionality
2. **Use Fixtures**: Reuse common test data and setup
3. **Mark Tests**: Use appropriate markers for test categories
4. **Keep Tests Fast**: Avoid slow operations in unit tests

### Maintenance
1. **Update Tests**: Keep tests in sync with code changes
2. **Remove Dead Tests**: Delete tests for removed functionality
3. **Refactor Tests**: Improve test quality over time
4. **Monitor Coverage**: Maintain good test coverage

## Troubleshooting

### Common Issues

#### Backend Tests
- **Import Errors**: Ensure PYTHONPATH includes the src directory
- **Mock Issues**: Check mock setup and return values
- **Database Issues**: Use in-memory databases for tests
- **Async Issues**: Use proper async/await patterns

#### Frontend Tests
- **Component Not Found**: Check import paths and component exports
- **Mock Not Working**: Verify mock setup and module paths
- **State Issues**: Reset state between tests
- **Timing Issues**: Use waitFor for async operations

### Debugging
- **Verbose Output**: Use `-v` flag for detailed output
- **Single Test**: Run individual tests for debugging
- **Print Statements**: Add temporary print statements for debugging
- **Test UI**: Use `npm run test:ui` for interactive testing

## Coverage Goals

- **Backend**: >90% code coverage
- **Frontend**: >85% code coverage
- **Critical Paths**: 100% coverage for core functionality
- **API Endpoints**: 100% coverage for all endpoints

## Performance Testing

### Backend Performance
- **Concurrent Requests**: Test multiple simultaneous API calls
- **Large Data**: Test with large repositories and datasets
- **Memory Usage**: Monitor memory consumption during tests
- **Response Times**: Measure API response times

### Frontend Performance
- **Rendering**: Test component rendering performance
- **State Updates**: Test state management performance
- **API Calls**: Test API call handling
- **User Interactions**: Test interaction responsiveness

## Security Testing

### Backend Security
- **Input Validation**: Test all input validation
- **Authentication**: Test authentication mechanisms
- **Authorization**: Test access control
- **Data Sanitization**: Test data cleaning and validation

### Frontend Security
- **XSS Prevention**: Test for cross-site scripting vulnerabilities
- **Input Sanitization**: Test user input handling
- **API Security**: Test API call security
- **Error Handling**: Test secure error handling

## Conclusion

This comprehensive testing suite ensures the RAG GitHub Assistant is reliable, maintainable, and ready for production use. The tests cover all major functionality, edge cases, and user workflows, providing confidence in the application's quality and stability.
