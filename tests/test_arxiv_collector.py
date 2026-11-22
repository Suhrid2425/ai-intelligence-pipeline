
from src.collectors.arxiv import ArxivCollector


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2601.12345v2</id>
        <title>  A Sample AI Research Paper  </title>
        <published>2026-01-15T10:30:00Z</published>
        <author>
            <name>Author One</name>
        </author>
        <author>
            <name>Author Two</name>
        </author>
    </entry>
</feed>
"""


def test_parse_arxiv_feed():
    collector = ArxivCollector()

    papers = collector.parse_feed(SAMPLE_ARXIV_XML)

    assert len(papers) == 1

    paper = papers[0]

    assert paper.content.title == "A Sample AI Research Paper"

    assert str(paper.content.paper_url) == (
    "https://arxiv.org/abs/2601.12345"
    )

    assert paper.content.authors == [
        "Author One",
        "Author Two",
    ]

    assert paper.content.github_url is None
    assert paper.content.github_stars is None

    assert paper.content.published_date.year == 2026


def test_checkpoint_save_and_load(tmp_path):
    collector = ArxivCollector()

    papers = collector.parse_feed(SAMPLE_ARXIV_XML)

    checkpoint_path = tmp_path / "arxiv_papers.jsonl"

    # Save the parsed papers.
    collector.save_checkpoint(
        papers={
            str(papers[0].content.paper_url): papers[0]
        },
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()

    # Load the saved checkpoint.
    loaded_papers = collector.load_checkpoint(
        checkpoint_path
    )

    assert len(loaded_papers) == 1

    loaded_paper = list(loaded_papers.values())[0]

    assert loaded_paper.content.title == (
        "A Sample AI Research Paper"
    )


def test_checkpoint_skips_invalid_lines(tmp_path):
    collector = ArxivCollector()

    checkpoint_path = tmp_path / "invalid_checkpoint.jsonl"

    checkpoint_path.write_text(
        '{"this": "is not a valid paper"}\n',
        encoding="utf-8",
    )

    loaded_papers = collector.load_checkpoint(
        checkpoint_path
    )

    assert loaded_papers == {}


def test_parse_empty_feed():
    collector = ArxivCollector()

    empty_xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    </feed>
    """

    papers = collector.parse_feed(empty_xml)

    assert papers == []
