
from src.collectors.github_enrichment import GitHubEnricher


def test_extract_github_repository():
    html = """
    <html>
        <body>
            <a href="https://github.com/example-owner/example-repo">
                GitHub repository
            </a>
        </body>
    </html>
    """

    result = GitHubEnricher.extract_github_repo_url(html)

    assert result == (
        "https://github.com/example-owner/example-repo"
    )


def test_extract_github_repository_with_git_suffix():
    html = """
    <html>
        <body>
            <a href="https://github.com/example-owner/example-repo.git">
                Source code
            </a>
        </body>
    </html>
    """

    result = GitHubEnricher.extract_github_repo_url(html)

    assert result == (
        "https://github.com/example-owner/example-repo"
    )


def test_ignore_non_github_links():
    html = """
    <html>
        <body>
            <a href="https://example.com/project">
                Project
            </a>
        </body>
    </html>
    """

    result = GitHubEnricher.extract_github_repo_url(html)

    assert result is None


def test_ignore_github_profile_links():
    html = """
    <html>
        <body>
            <a href="https://github.com/example-owner">
                GitHub profile
            </a>
        </body>
    </html>
    """

    result = GitHubEnricher.extract_github_repo_url(html)

    assert result is None


def test_ignore_github_topics():
    html = """
    <html>
        <body>
            <a href="https://github.com/topics/machine-learning">
                Topic
            </a>
        </body>
    </html>
    """

    result = GitHubEnricher.extract_github_repo_url(html)

    assert result is None


def test_empty_html():
    result = GitHubEnricher.extract_github_repo_url("")

    assert result is None
