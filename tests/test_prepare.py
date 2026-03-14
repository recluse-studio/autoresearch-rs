import unittest

from prepare import extract_visible_prose, is_eligible_markdown


class PrepareTests(unittest.TestCase):
    def test_extract_visible_prose_keeps_links_and_drops_code(self) -> None:
        markdown = """---
title: Sample
---

# Heading

See [this link](https://example.com).

```python
print("ignore me")
```

Inline `code` should vanish.
"""
        segments = extract_visible_prose(markdown)
        joined = "\n".join(segments)
        self.assertIn("Heading", joined)
        self.assertIn("this link", joined)
        self.assertNotIn("print(", joined)
        self.assertNotIn("code", joined)

    def test_markdown_eligibility_filters_program_and_backups(self) -> None:
        self.assertFalse(is_eligible_markdown("program.md"))
        self.assertFalse(is_eligible_markdown("karpathy-files/README.md"))
        self.assertTrue(is_eligible_markdown("chapters/intro.md"))


if __name__ == "__main__":
    unittest.main()
