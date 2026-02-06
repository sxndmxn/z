use crate::structs::{Result, XmlElement, ZError};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::cell::RefCell;
use std::fs;
use std::io::Cursor;
use std::path::Path;

/// Size limits for LLM tool responses
pub const MAX_XML_ELEMENTS: usize = 10;

/// XML modifier that can query and modify XML files
pub struct XmlModifier {
    content: RefCell<String>,
}

impl XmlModifier {
    /// Load XML from a file
    ///
    /// # Errors
    /// Returns error if file cannot be read
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(Self {
            content: RefCell::new(content),
        })
    }

    /// Load XML from a string
    #[allow(dead_code)]
    #[must_use]
    pub const fn from_string(content: String) -> Self {
        Self {
            content: RefCell::new(content),
        }
    }

    /// Get current XML content
    #[must_use]
    pub fn get_content(&self) -> String {
        self.content.borrow().clone()
    }

    /// Get the XML structure as a hierarchy
    ///
    /// # Errors
    /// Returns error if XML parsing fails
    pub fn get_structure(&self) -> Result<Vec<XmlElement>> {
        let content = self.content.borrow();
        let mut reader = Reader::from_str(&content);
        reader.trim_text(true);

        let mut elements = Vec::new();
        let mut path_stack: Vec<String> = Vec::new();

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());
                    let path = path_stack.join("/");

                    let attributes: Vec<(String, String)> = e
                        .attributes()
                        .filter_map(std::result::Result::ok)
                        .map(|a| {
                            let key = String::from_utf8_lossy(a.key.as_ref()).to_string();
                            let value = String::from_utf8_lossy(&a.value).to_string();
                            (key, value)
                        })
                        .collect();

                    elements.push(XmlElement {
                        path: path.clone(),
                        name,
                        attributes,
                        text: None,
                        depth: path_stack.len() - 1,
                    });
                }
                Ok(Event::Text(e)) => {
                    let text = e.unescape().unwrap_or_default().trim().to_string();
                    if !text.is_empty() {
                        if let Some(last) = elements.last_mut() {
                            last.text = Some(text);
                        }
                    }
                }
                Ok(Event::End(_)) => {
                    path_stack.pop();
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());
                    let path = path_stack.join("/");

                    let attributes: Vec<(String, String)> = e
                        .attributes()
                        .filter_map(std::result::Result::ok)
                        .map(|a| {
                            let key = String::from_utf8_lossy(a.key.as_ref()).to_string();
                            let value = String::from_utf8_lossy(&a.value).to_string();
                            (key, value)
                        })
                        .collect();

                    elements.push(XmlElement {
                        path,
                        name,
                        attributes,
                        text: None,
                        depth: path_stack.len() - 1,
                    });

                    path_stack.pop();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(ZError::Xml(e)),
                _ => {}
            }
        }

        Ok(elements)
    }

    /// Query elements matching a simplified path pattern
    /// Supports: `parent/child`, `element[@attr='value']`
    ///
    /// # Errors
    /// Returns error if XML parsing fails
    pub fn query(&self, pattern: &str) -> Result<Vec<XmlElement>> {
        let elements = self.get_structure()?;

        let (path_pattern, attr_filter) = parse_pattern(pattern);

        let matched: Vec<XmlElement> = elements
            .into_iter()
            .filter(|e| {
                if !path_matches(&e.path, &e.name, &path_pattern) {
                    return false;
                }

                // Match attribute filter if present
                if let Some((attr_name, attr_value)) = &attr_filter {
                    e.attributes
                        .iter()
                        .any(|(k, v)| k == attr_name && v == attr_value)
                } else {
                    true
                }
            })
            .take(MAX_XML_ELEMENTS)
            .collect();

        Ok(matched)
    }

    /// Get a specific element by exact path
    ///
    /// # Errors
    /// Returns error if XML parsing fails
    pub fn get_element(&self, path: &str) -> Result<Option<XmlElement>> {
        let elements = self.get_structure()?;
        Ok(elements.into_iter().find(|e| e.path == path))
    }

    /// Update text content of an element matching the path
    ///
    /// # Errors
    /// Returns error if XML parsing or modification fails
    pub fn update_text(&self, path_pattern: &str, new_text: &str) -> Result<bool> {
        let (path_pattern, attr_filter) = parse_pattern(path_pattern);
        let content = self.content.borrow().clone();
        let mut reader = Reader::from_str(&content);
        reader.trim_text(false);

        let mut writer = Writer::new(Cursor::new(Vec::new()));
        let mut path_stack: Vec<String> = Vec::new();
        let mut modified = false;
        let mut in_target = false;

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    let current_path = path_stack.join("/");
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, attr_filter.as_ref());

                    in_target = matches_path && attr_matches && !modified;
                    writer.write_event(Event::Start(e))?;
                }
                Ok(Event::Text(e)) => {
                    if in_target && !modified {
                        writer.write_event(Event::Text(BytesText::new(new_text)))?;
                        modified = true;
                    } else {
                        writer.write_event(Event::Text(e))?;
                    }
                }
                Ok(Event::End(e)) => {
                    // If we were in target but never saw text, insert it
                    if in_target && !modified {
                        writer.write_event(Event::Text(BytesText::new(new_text)))?;
                        modified = true;
                    }
                    in_target = false;
                    path_stack.pop();
                    writer.write_event(Event::End(e))?;
                }
                Ok(Event::Eof) => break,
                Ok(e) => writer.write_event(e)?,
                Err(e) => return Err(ZError::Xml(e)),
            }
        }

        if modified {
            let new_content = finish_writer(writer)?;
            *self.content.borrow_mut() = new_content;
        }

        Ok(modified)
    }

    /// Set an attribute on an element matching the path
    ///
    /// # Errors
    /// Returns error if XML parsing or modification fails
    pub fn set_attribute(
        &self,
        path_pattern: &str,
        attr_name: &str,
        attr_value: &str,
    ) -> Result<bool> {
        let (path_pattern, existing_filter) = parse_pattern(path_pattern);
        let content = self.content.borrow().clone();
        let mut reader = Reader::from_str(&content);
        reader.trim_text(false);

        let mut writer = Writer::new(Cursor::new(Vec::new()));
        let mut path_stack: Vec<String> = Vec::new();
        let mut modified = false;

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    let current_path = path_stack.join("/");
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, existing_filter.as_ref());

                    if matches_path && attr_matches && !modified {
                        let new_elem =
                            build_element_with_attr(&e, &name, attr_name, attr_value);
                        writer.write_event(Event::Start(new_elem))?;
                        modified = true;
                    } else {
                        writer.write_event(Event::Start(e))?;
                    }
                }
                Ok(Event::End(e)) => {
                    path_stack.pop();
                    writer.write_event(Event::End(e))?;
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    let current_path = path_stack.join("/");
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, existing_filter.as_ref());

                    if matches_path && attr_matches && !modified {
                        let new_elem =
                            build_element_with_attr(&e, &name, attr_name, attr_value);
                        writer.write_event(Event::Empty(new_elem))?;
                        modified = true;
                    } else {
                        writer.write_event(Event::Empty(e))?;
                    }

                    path_stack.pop();
                }
                Ok(Event::Eof) => break,
                Ok(e) => writer.write_event(e)?,
                Err(e) => return Err(ZError::Xml(e)),
            }
        }

        if modified {
            let new_content = finish_writer(writer)?;
            *self.content.borrow_mut() = new_content;
        }

        Ok(modified)
    }

    /// Delete an element matching the path
    ///
    /// # Errors
    /// Returns error if XML parsing or modification fails
    pub fn delete_element(&self, path_pattern: &str) -> Result<bool> {
        let (path_pattern, attr_filter) = parse_pattern(path_pattern);
        let content = self.content.borrow().clone();
        let mut reader = Reader::from_str(&content);
        reader.trim_text(false);

        let mut writer = Writer::new(Cursor::new(Vec::new()));
        let mut path_stack: Vec<String> = Vec::new();
        let mut modified = false;
        let mut skip_depth: Option<usize> = None;

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    // If we're already skipping, continue
                    if skip_depth.is_some() {
                        continue;
                    }

                    let current_path = path_stack.join("/");
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, attr_filter.as_ref());

                    if matches_path && attr_matches && !modified {
                        skip_depth = Some(path_stack.len());
                        modified = true;
                    } else {
                        writer.write_event(Event::Start(e))?;
                    }
                }
                Ok(Event::End(e)) => {
                    let depth = path_stack.len();
                    path_stack.pop();

                    if let Some(skip_at) = skip_depth {
                        if depth == skip_at {
                            skip_depth = None;
                        }
                        continue;
                    }

                    writer.write_event(Event::End(e))?;
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    if skip_depth.is_none() {
                        let current_path = path_stack.join("/");
                        let matches_path = path_matches(&current_path, &name, &path_pattern);
                        let attr_matches = check_attr_filter(&e, attr_filter.as_ref());

                        if matches_path && attr_matches && !modified {
                            modified = true;
                        } else {
                            writer.write_event(Event::Empty(e))?;
                        }
                    }

                    path_stack.pop();
                }
                Ok(Event::Text(e)) => {
                    if skip_depth.is_none() {
                        writer.write_event(Event::Text(e))?;
                    }
                }
                Ok(Event::Eof) => break,
                Ok(e) => {
                    if skip_depth.is_none() {
                        writer.write_event(e)?;
                    }
                }
                Err(e) => return Err(ZError::Xml(e)),
            }
        }

        if modified {
            let new_content = finish_writer(writer)?;
            *self.content.borrow_mut() = new_content;
        }

        Ok(modified)
    }

    /// Insert a new element as a child of the matching parent
    ///
    /// # Errors
    /// Returns error if XML parsing or modification fails
    pub fn insert_element(
        &self,
        parent_pattern: &str,
        element_name: &str,
        attributes: &[(String, String)],
        text: Option<&str>,
    ) -> Result<bool> {
        let (path_pattern, attr_filter) = parse_pattern(parent_pattern);
        let content = self.content.borrow().clone();
        let mut reader = Reader::from_str(&content);
        reader.trim_text(false);

        let mut writer = Writer::new(Cursor::new(Vec::new()));
        let mut path_stack: Vec<String> = Vec::new();
        let mut modified = false;
        let mut target_depth: Option<usize> = None;

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    path_stack.push(name.clone());

                    let current_path = path_stack.join("/");
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, attr_filter.as_ref());

                    if matches_path && attr_matches && !modified {
                        target_depth = Some(path_stack.len());
                    }

                    writer.write_event(Event::Start(e))?;
                }
                Ok(Event::End(e)) => {
                    let depth = path_stack.len();

                    // Insert before closing the target element
                    if target_depth == Some(depth) && !modified {
                        write_new_element(&mut writer, element_name, attributes, text)?;
                        modified = true;
                        target_depth = None;
                    }

                    path_stack.pop();
                    writer.write_event(Event::End(e))?;
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    // For empty parent elements, expand them
                    let current_path = format!("{}/{name}", path_stack.join("/"));
                    let matches_path = path_matches(&current_path, &name, &path_pattern);
                    let attr_matches = check_attr_filter(&e, attr_filter.as_ref());

                    if matches_path && attr_matches && !modified {
                        // Convert empty to start tag
                        let start = BytesStart::new(&name);
                        writer.write_event(Event::Start(start))?;

                        // Add new element
                        write_new_element(&mut writer, element_name, attributes, text)?;

                        writer.write_event(Event::End(BytesEnd::new(&name)))?;
                        modified = true;
                    } else {
                        writer.write_event(Event::Empty(e))?;
                    }
                }
                Ok(Event::Eof) => break,
                Ok(e) => writer.write_event(e)?,
                Err(e) => return Err(ZError::Xml(e)),
            }
        }

        if modified {
            let new_content = finish_writer(writer)?;
            *self.content.borrow_mut() = new_content;
        }

        Ok(modified)
    }

    /// Write to a file atomically (write to .tmp, then rename)
    ///
    /// # Errors
    /// Returns error if file operations fail
    pub fn write_to_file(content: &str, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("xml.tmp");
        fs::write(&tmp_path, content)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

/// Check if the current element path matches a pattern
fn path_matches(current_path: &str, name: &str, pattern: &str) -> bool {
    if pattern.contains('/') {
        current_path.ends_with(pattern) || current_path == pattern
    } else {
        current_path.ends_with(pattern) || current_path == pattern || name == pattern
    }
}

/// Parse a path pattern like `element[@attr='value']`
fn parse_pattern(pattern: &str) -> (String, Option<(String, String)>) {
    if let Some(bracket_start) = pattern.find("[@") {
        if let Some(bracket_end) = pattern.find(']') {
            let path = pattern[..bracket_start].to_string();
            let attr_part = &pattern[bracket_start + 2..bracket_end];

            if let Some(eq_pos) = attr_part.find('=') {
                let attr_name = attr_part[..eq_pos].to_string();
                let attr_value = attr_part[eq_pos + 1..]
                    .trim_matches('\'')
                    .trim_matches('"')
                    .to_string();
                return (path, Some((attr_name, attr_value)));
            }
        }
    }

    (pattern.to_string(), None)
}

/// Check if element matches the attribute filter
fn check_attr_filter(e: &BytesStart<'_>, filter: Option<&(String, String)>) -> bool {
    if let Some((filter_name, filter_value)) = filter {
        e.attributes()
            .filter_map(std::result::Result::ok)
            .any(|a| {
                let key = String::from_utf8_lossy(a.key.as_ref());
                let val = String::from_utf8_lossy(&a.value);
                key == *filter_name && val == *filter_value
            })
    } else {
        true
    }
}

/// Build a new element with an attribute set/updated
fn build_element_with_attr<'a>(
    original: &BytesStart<'_>,
    name: &'a str,
    attr_name: &str,
    attr_value: &str,
) -> BytesStart<'a> {
    let mut new_elem = BytesStart::new(name);

    let mut found_attr = false;
    for attr in original.attributes().filter_map(std::result::Result::ok) {
        let key = String::from_utf8_lossy(attr.key.as_ref());
        if key == attr_name {
            new_elem.push_attribute((attr_name, attr_value));
            found_attr = true;
        } else {
            new_elem.push_attribute(attr);
        }
    }

    if !found_attr {
        new_elem.push_attribute((attr_name, attr_value));
    }

    new_elem
}

/// Write a new element to the writer
fn write_new_element<W: std::io::Write>(
    writer: &mut Writer<W>,
    element_name: &str,
    attributes: &[(String, String)],
    text: Option<&str>,
) -> Result<()> {
    writer.write_event(Event::Text(BytesText::new("\n    ")))?;

    let mut elem = BytesStart::new(element_name);
    for (key, val) in attributes {
        elem.push_attribute((key.as_str(), val.as_str()));
    }

    if let Some(txt) = text {
        writer.write_event(Event::Start(elem))?;
        writer.write_event(Event::Text(BytesText::new(txt)))?;
        writer.write_event(Event::End(BytesEnd::new(element_name)))?;
    } else {
        writer.write_event(Event::Empty(elem))?;
    }

    writer.write_event(Event::Text(BytesText::new("\n  ")))?;
    Ok(())
}

/// Finish writing and convert to string
fn finish_writer(writer: Writer<Cursor<Vec<u8>>>) -> Result<String> {
    let result = writer.into_inner().into_inner();
    String::from_utf8(result).map_err(|e| {
        ZError::Xml(quick_xml::Error::Io(std::sync::Arc::new(
            std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_structure() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <items>
    <item id="1">First</item>
    <item id="2">Second</item>
  </items>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());
        let structure = modifier.get_structure().expect("parse structure");

        assert!(structure.iter().any(|e| e.path == "root"));
        assert!(structure.iter().any(|e| e.path == "root/items"));
        assert!(structure.iter().any(|e| e.path == "root/items/item"));
    }

    #[test]
    fn test_query() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <item id="1">First</item>
  <item id="2">Second</item>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());

        let items = modifier.query("item").expect("query");
        assert_eq!(items.len(), 2);

        let item1 = modifier.query("item[@id='1']").expect("query");
        assert_eq!(item1.len(), 1);
        assert_eq!(item1[0].text.as_deref(), Some("First"));
    }

    #[test]
    fn test_update_text() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <name>Old</name>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());
        let modified = modifier.update_text("name", "New").expect("update");

        assert!(modified);
        assert!(modifier.get_content().contains("New"));
    }

    #[test]
    fn test_set_attribute() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <item id="1">Test</item>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());
        let modified = modifier
            .set_attribute("item[@id='1']", "status", "active")
            .expect("set attr");

        assert!(modified);
        assert!(modifier.get_content().contains("status=\"active\""));
    }

    #[test]
    fn test_delete_element() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <item id="1">Keep</item>
  <item id="2">Delete</item>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());
        let modified = modifier.delete_element("item[@id='2']").expect("delete");

        assert!(modified);
        let content = modifier.get_content();
        assert!(content.contains("Keep"));
        assert!(!content.contains("Delete"));
    }

    #[test]
    fn test_insert_element() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <items>
  </items>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());
        let modified = modifier
            .insert_element(
                "items",
                "item",
                &[("id".to_string(), "new".to_string())],
                Some("New item"),
            )
            .expect("insert");

        assert!(modified);
        let content = modifier.get_content();
        assert!(content.contains("<item id=\"new\">New item</item>"));
    }

    #[test]
    fn test_parse_pattern() {
        let (path, filter) = parse_pattern("item[@id='123']");
        assert_eq!(path, "item");
        assert_eq!(filter, Some(("id".to_string(), "123".to_string())));

        let (path, filter) = parse_pattern("root/items/item");
        assert_eq!(path, "root/items/item");
        assert!(filter.is_none());
    }
}
