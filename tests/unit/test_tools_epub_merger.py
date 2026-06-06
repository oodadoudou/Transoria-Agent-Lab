from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from transoria.tools.epub_merger import (
    EpubMergeAction,
    EpubMergeOptions,
    build_epub_merge_plan,
    merge_epub_files,
)


def test_build_epub_merge_plan_sorts_and_defaults_output_to_input_folder(tmp_path: Path) -> None:
    _write_epub(tmp_path / "Novel 2화.epub", title="Internal 2", chapter="Two")
    _write_epub(tmp_path / "Novel 1화.epub", title="Internal 1", chapter="One")

    plan = build_epub_merge_plan(
        tmp_path,
        options=EpubMergeOptions(),
    )

    assert [Path(action.source_path).name for action in plan.actions] == [
        "Novel 1화.epub",
        "Novel 2화.epub",
    ]
    assert plan.output_path.name == "merged.epub"
    assert plan.output_path.parent == tmp_path.resolve()


def test_build_epub_merge_plan_sorts_main_volumes_before_side_stories(tmp_path: Path) -> None:
    names = [
        "[플로나] 모두가 그대를 증오할지라도 1권.epub",
        "[플로나] 모두가 그대를 증오할지라도 2권.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 2.epub",
        "[플로나] 모두가 그대를 증오할지라도 3권.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 1.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 3.epub",
        "[플로나] 모두가 그대를 증오할지라도 4권.epub",
        "[플로나] 모두가 그대를 증오할지라도 5권.epub",
    ]
    for name in names:
        _write_epub(tmp_path / name, title=name, chapter=name)

    plan = build_epub_merge_plan(tmp_path, options=EpubMergeOptions())

    assert [Path(action.source_path).name for action in plan.actions] == [
        "[플로나] 모두가 그대를 증오할지라도 1권.epub",
        "[플로나] 모두가 그대를 증오할지라도 2권.epub",
        "[플로나] 모두가 그대를 증오할지라도 3권.epub",
        "[플로나] 모두가 그대를 증오할지라도 4권.epub",
        "[플로나] 모두가 그대를 증오할지라도 5권.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 1.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 2.epub",
        "[플로나] 모두가 그대를 증오할지라도 외전 3.epub",
    ]


def test_build_epub_merge_plan_uses_natural_sort_for_chinese_variants(tmp_path: Path) -> None:
    names = [
        "作品 外傳 2.epub",
        "作品 第10卷.epub",
        "作品 第2卷.epub",
        "作品 外传 1.epub",
        "作品 第1卷.epub",
    ]
    for name in names:
        _write_epub(tmp_path / name, title=name, chapter=name)

    plan = build_epub_merge_plan(tmp_path, options=EpubMergeOptions())

    assert [Path(action.source_path).name for action in plan.actions] == [
        "作品 第1卷.epub",
        "作品 第2卷.epub",
        "作品 第10卷.epub",
        "作品 外传 1.epub",
        "作品 外傳 2.epub",
    ]


def test_merge_uses_output_filename_for_metadata_title(tmp_path: Path) -> None:
    first = tmp_path / "Novel 1화.epub"
    second = tmp_path / "Novel 2화.epub"
    output = tmp_path / "通奸 7.epub"
    _write_epub(first, title="Volume 1", chapter="First", author="Author A")
    _write_epub(second, title="Volume 2", chapter="Second", author="Author B")

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(first, second),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist()[0] == "mimetype"
        opf = archive.read("content.opf").decode("utf-8")
        assert "<dc:title>通奸 7</dc:title>" in opf
        assert 'properties="cover-image"' in opf
        assert '<meta name="cover" content="img_000_0"/>' in opf
        assert "font.ttf" not in opf
        assert "epub_000/Volume 1.xhtml" in archive.namelist()
        assert "epub_001/Volume 2.xhtml" in archive.namelist()
        nav = archive.read("nav.xhtml").decode("utf-8")
        assert "Volume 1" in nav
        assert "Volume 2" in nav


def test_merge_single_epub_rewrites_metadata_title(tmp_path: Path) -> None:
    source = tmp_path / "Novel 1화.epub"
    output = tmp_path / "Renamed Novel.epub"
    _write_epub(source, title="Old Title", chapter="Only Chapter", author="Author A")

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    assert result.merged_files == 1
    with zipfile.ZipFile(output) as archive:
        opf = archive.read("content.opf").decode("utf-8")
        assert "<dc:title>Renamed Novel</dc:title>" in opf
        assert "<dc:creator>Author A</dc:creator>" in opf
        nav = archive.read("nav.xhtml").decode("utf-8")
        assert "Only Chapter" in nav


def test_merge_single_epub_keeps_chapter_level_navigation(tmp_path: Path) -> None:
    source = tmp_path / "Novel 1화.epub"
    output = tmp_path / "Renamed Novel.epub"
    _write_epub(
        source,
        title="Old Title",
        chapter="Chapter 1",
        extra_html={
            "chapter2.xhtml": "<h1>Chapter 2</h1>",
            "chapter3.xhtml": "<h1>Chapter 3</h1>",
        },
        ncx_body="""
<navPoint id="nav-1" playOrder="1">
  <navLabel><text>Old Title</text></navLabel>
  <content src="chapter.xhtml"/>
  <navPoint id="nav-1-1" playOrder="2">
    <navLabel><text>Chapter 1</text></navLabel>
    <content src="chapter.xhtml"/>
  </navPoint>
  <navPoint id="nav-1-2" playOrder="3">
    <navLabel><text>Chapter 2</text></navLabel>
    <content src="chapter2.xhtml"/>
  </navPoint>
  <navPoint id="nav-1-3" playOrder="4">
    <navLabel><text>Chapter 3</text></navLabel>
    <content src="chapter3.xhtml"/>
  </navPoint>
</navPoint>
""",
    )

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    assert result.chapters_written == 3
    with zipfile.ZipFile(output) as archive:
        opf = archive.read("content.opf").decode("utf-8")
        nav = archive.read("nav.xhtml").decode("utf-8")

    assert opf.count("<itemref") == 3
    assert "<dc:title>Renamed Novel</dc:title>" in opf
    assert nav.count("<li><a") == 3
    assert ">Old Title<" not in nav


def test_merge_skips_single_generic_child_toc_and_uses_source_titles(tmp_path: Path) -> None:
    first = tmp_path / "1.epub"
    second = tmp_path / "2.epub"
    output = tmp_path / "通奸期.epub"
    _write_epub(first, title="1", chapter="Section0001", toc_title="불륜기")
    _write_epub(second, title="2", chapter="Section0001", toc_title="불륜기")

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(first, second),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        nav = archive.read("nav.xhtml").decode("utf-8")
        assert ">1<" in nav
        assert ">2<" in nav
        assert "Section0001" not in nav
        assert "불륜기" not in nav


def test_merge_collapses_duplicate_nested_ncx_entries(tmp_path: Path) -> None:
    source = tmp_path / "Volume.epub"
    output = tmp_path / "merged.epub"
    _write_epub(
        source,
        title="Volume",
        chapter="Bermuda 350화",
        extra_html={"chapter351.xhtml": "<h1>Bermuda 351화</h1>"},
        ncx_body="""
<navPoint id="nav-1" playOrder="1">
  <navLabel><text>Bermuda 350화</text></navLabel>
  <content src="chapter.xhtml"/>
  <navPoint id="nav-1-1" playOrder="2">
    <navLabel><text>Bermuda 350화</text></navLabel>
    <content src="chapter.xhtml"/>
    </navPoint>
  </navPoint>
<navPoint id="nav-2" playOrder="3">
  <navLabel><text>Bermuda 351화</text></navLabel>
  <content src="chapter351.xhtml"/>
</navPoint>
""",
    )

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        nav = archive.read("nav.xhtml").decode("utf-8")
        ncx = archive.read("toc.ncx").decode("utf-8")

    assert nav.count("Bermuda 350화") == 1
    assert "Bermuda 350화Bermuda 350화" not in ncx
    assert ncx.count("<text>Bermuda 350화</text>") == 1


def test_merge_collapses_cover_parent_to_content_nav_entry(tmp_path: Path) -> None:
    source = tmp_path / "Volume.epub"
    output = tmp_path / "merged.epub"
    _write_epub(
        source,
        title="Volume",
        chapter="Bermuda 353화",
        include_cover_page=True,
        extra_html={"chapter354.xhtml": "<h1>Bermuda 354화</h1>"},
        ncx_body="""
<navPoint id="nav-1" playOrder="1">
  <navLabel><text>Bermuda 353화</text></navLabel>
  <content src="auto_cover.xhtml"/>
  <navPoint id="nav-1-1" playOrder="2">
    <navLabel><text>표지</text></navLabel>
    <content src="auto_cover.xhtml"/>
  </navPoint>
  <navPoint id="nav-1-2" playOrder="3">
    <navLabel><text>Bermuda 353화</text></navLabel>
    <content src="chapter.xhtml"/>
  </navPoint>
</navPoint>
<navPoint id="nav-2" playOrder="4">
  <navLabel><text>Bermuda 354화</text></navLabel>
  <content src="chapter354.xhtml"/>
</navPoint>
""",
    )

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        nav = archive.read("nav.xhtml").decode("utf-8")
        ncx = archive.read("toc.ncx").decode("utf-8")

    assert nav.count("Bermuda 353화") == 1
    assert "표지" not in nav
    assert "auto_cover" not in nav
    assert "epub_000/Volume.xhtml" in nav
    assert ncx.count("<text>Bermuda 353화</text>") == 1


def test_merge_preserves_real_nested_toc_entries(tmp_path: Path) -> None:
    source = tmp_path / "Volume.epub"
    output = tmp_path / "merged.epub"
    _write_epub(
        source,
        title="Volume",
        chapter="Opening Scene",
        extra_html={
            "chapter2.xhtml": "<h1>Second Scene</h1>",
            "part2.xhtml": "<h1>Afterword</h1>",
        },
        ncx_body="""
<navPoint id="nav-1" playOrder="1">
  <navLabel><text>Main Arc</text></navLabel>
  <content src="chapter.xhtml"/>
  <navPoint id="nav-1-1" playOrder="2">
    <navLabel><text>Opening Scene</text></navLabel>
    <content src="chapter.xhtml"/>
  </navPoint>
  <navPoint id="nav-1-2" playOrder="3">
    <navLabel><text>Second Scene</text></navLabel>
    <content src="chapter2.xhtml"/>
  </navPoint>
</navPoint>
<navPoint id="nav-2" playOrder="4">
  <navLabel><text>Afterword</text></navLabel>
  <content src="part2.xhtml"/>
</navPoint>
""",
    )

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        nav = archive.read("nav.xhtml").decode("utf-8")

    assert "Main Arc" in nav
    assert "Opening Scene" in nav
    assert "Second Scene" in nav
    assert "Afterword" in nav
    assert "<ol><li><a" in nav


def test_merge_rewrites_css_and_image_links_and_deduplicates_images(tmp_path: Path) -> None:
    image = _image_bytes(color=(100, 80, 60))
    first = tmp_path / "Novel 1화.epub"
    second = tmp_path / "Novel 2화.epub"
    output = tmp_path / "merged.epub"
    _write_epub(first, title="Novel", chapter="First", image=image)
    _write_epub(second, title="Novel", chapter="Second", image=image)

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(first, second),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    assert result.images_deduplicated == 1
    with zipfile.ZipFile(output) as archive:
        images = [name for name in archive.namelist() if name.startswith("images/")]
        assert len(images) == 1
        first_html = next(name for name in archive.namelist() if name.startswith("epub_000/") and name.endswith(".xhtml"))
        html = archive.read(first_html).decode("utf-8")
        css = archive.read("styles/000_style.css").decode("utf-8")
        assert "../images/" in html
        assert "url(\"../images/" in css
        assert "@font-face" not in css


def test_merge_strips_orphan_markup_before_xhtml_document(tmp_path: Path) -> None:
    source = tmp_path / "Novel 1화.epub"
    output = tmp_path / "merged.epub"
    _write_epub(
        source,
        title="Novel",
        chapter="First",
        chapter_prefix='<img alt="图片" src="../dropped_image.png" />',
    )

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=output,
        actions=_actions(source),
        options=EpubMergeOptions(),
    )

    assert result.status == "merged"
    with zipfile.ZipFile(output) as archive:
        first_html = next(name for name in archive.namelist() if name.startswith("epub_000/") and name.endswith(".xhtml"))
        html = archive.read(first_html).decode("utf-8")

    assert html.startswith('<?xml version="1.0"')
    assert "dropped_image" not in html
    assert "<h1>First</h1>" in html


def test_merge_disallows_overwriting_selected_input(tmp_path: Path) -> None:
    first = tmp_path / "Novel 1화.epub"
    second = tmp_path / "Novel 2화.epub"
    _write_epub(first, title="Novel", chapter="First")
    _write_epub(second, title="Novel", chapter="Second")

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=first,
        actions=_actions(first, second),
        options=EpubMergeOptions(),
    )

    assert result.status == "failed"
    assert "overwrite" in result.error


def test_merge_skips_bad_epub_but_fails_when_no_spine_is_left(tmp_path: Path) -> None:
    bad = tmp_path / "bad.epub"
    other = tmp_path / "other.epub"
    bad.write_text("not zip", encoding="utf-8")
    other.write_text("not zip", encoding="utf-8")

    result = merge_epub_files(
        action_id="merge-0000",
        input_dir=tmp_path,
        output_path=tmp_path / "merged.epub",
        actions=_actions(bad, other),
        options=EpubMergeOptions(),
    )

    assert result.status == "failed"
    assert "no readable spine" in result.error


def _actions(*paths: Path) -> list[EpubMergeAction]:
    return [
        EpubMergeAction(
            id=f"epub-{index:04d}",
            source_path=str(path),
            order=index,
            title_hint=path.stem,
            size_bytes=path.stat().st_size,
        )
        for index, path in enumerate(paths)
    ]


def _write_epub(
    path: Path,
    *,
    title: str,
    chapter: str,
    author: str = "Author",
    image: bytes | None = None,
    toc_title: str | None = None,
    chapter_prefix: str = "",
    ncx_body: str | None = None,
    include_cover_page: bool = False,
    extra_html: dict[str, str] | None = None,
) -> None:
    image = image or _image_bytes(color=(120, 80, 40))
    extra_html = extra_html or {}
    extra_items = "\n".join(
        f'    <item id="extra_{index}" href="{href}" media-type="application/xhtml+xml"/>'
        for index, href in enumerate(extra_html, start=1)
    )
    extra_spine = "\n".join(
        f'    <itemref idref="extra_{index}"/>'
        for index, _href in enumerate(extra_html, start=1)
    )
    cover_item = (
        '    <item id="cover_page" href="auto_cover.xhtml" media-type="application/xhtml+xml"/>\n'
        if include_cover_page
        else ""
    )
    cover_spine = '    <itemref idref="cover_page"/>\n' if include_cover_page else ""
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
  <metadata>
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:language>ko</dc:language>
    <meta name="cover" content="img"/>
  </metadata>
  <manifest>
{cover_item}
    <item id="chap" href="chapter.xhtml" media-type="application/xhtml+xml"/>
{extra_items}
    <item id="style" href="style.css" media-type="text/css"/>
    <item id="img" href="Images/pic.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="font" href="Fonts/font.ttf" media-type="application/x-font-ttf"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
{cover_spine}
    <itemref idref="chap"/>
{extra_spine}
  </spine>
</package>"""
    html = f"""{chapter_prefix}<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>{chapter}</title><link href="style.css" rel="stylesheet"/></head>
<body><h1>{chapter}</h1><img src="Images/pic.jpg"/></body>
</html>"""
    css = "@font-face { src: url('Fonts/font.ttf'); }\nbody { background: url('Images/pic.jpg'); }"
    toc_label = toc_title or chapter
    ncx_body = ncx_body or (
        f'<navPoint id="nav-1" playOrder="1"><navLabel><text>{toc_label}</text></navLabel><content src="chapter.xhtml"/></navPoint>'
    )
    ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="test"/></head>
<docTitle><text>{title}</text></docTitle>
<navMap>
{ncx_body}
</navMap>
</ncx>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""",
        )
        archive.writestr("OEBPS/content.opf", opf)
        if include_cover_page:
            archive.writestr(
                "OEBPS/auto_cover.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><img src="Images/pic.jpg"/></body></html>""",
            )
        archive.writestr("OEBPS/chapter.xhtml", html)
        for href, body in extra_html.items():
            archive.writestr(
                f"OEBPS/{href}",
                f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{href}</title></head><body>{body}</body></html>""",
            )
        archive.writestr("OEBPS/style.css", css)
        archive.writestr("OEBPS/toc.ncx", ncx)
        archive.writestr("OEBPS/Images/pic.jpg", image)
        archive.writestr("OEBPS/Fonts/font.ttf", b"font")


def _image_bytes(*, color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (128, 128), color=color)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()
