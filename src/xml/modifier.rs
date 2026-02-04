use crate::db::DataRow;
use crate::error::{Result, ZError};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde_json::Value;
use std::fs;
use std::io::Cursor;
use std::path::Path;

/// XML modifier that can insert data rows into an XML file
pub struct XmlModifier {
    content: String,
}

impl XmlModifier {
    /// Load XML from a file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ZError::Io(e))?;
        Ok(XmlModifier { content })
    }

    /// Load XML from a string
    #[allow(dead_code)]
    pub fn from_string(content: String) -> Self {
        XmlModifier { content }
    }

    /// Insert data rows into the XML at the specified parent element
    /// Creates new child elements for each row
    pub fn insert_rows(
        &self,
        rows: &[DataRow],
        parent_element: &str,
        element_name: &str,
    ) -> Result<String> {
        let mut reader = Reader::from_str(&self.content);
        reader.trim_text(false);

        let mut writer = Writer::new(Cursor::new(Vec::new()));
        let mut depth = 0;
        let mut in_target = false;
        let mut target_depth = 0;
        let mut inserted = false;

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if name == parent_element && !in_target {
                        in_target = true;
                        target_depth = depth;
                    }

                    writer.write_event(Event::Start(e.clone()))?;
                    depth += 1;
                }
                Ok(Event::End(e)) => {
                    depth -= 1;
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    // Insert rows before closing parent element
                    if in_target && depth == target_depth && name == parent_element && !inserted {
                        self.write_rows(&mut writer, rows, element_name)?;
                        inserted = true;
                        in_target = false;
                    }

                    writer.write_event(Event::End(e))?;
                }
                Ok(Event::Empty(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    // Handle self-closing parent element by expanding it
                    if name == parent_element && !inserted {
                        // Convert to start tag
                        let start = BytesStart::new(&name);
                        writer.write_event(Event::Start(start))?;

                        // Write rows
                        self.write_rows(&mut writer, rows, element_name)?;

                        // Close tag
                        let end = BytesEnd::new(&name);
                        writer.write_event(Event::End(end))?;
                        inserted = true;
                    } else {
                        writer.write_event(Event::Empty(e))?;
                    }
                }
                Ok(Event::Eof) => break,
                Ok(e) => writer.write_event(e)?,
                Err(e) => return Err(ZError::Xml(e)),
            }
        }

        if !inserted {
            return Err(ZError::Xml(quick_xml::Error::Io(std::sync::Arc::new(
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Parent element '{}' not found", parent_element),
                ),
            ))));
        }

        let result = writer.into_inner().into_inner();
        String::from_utf8(result)
            .map_err(|e| ZError::Xml(quick_xml::Error::Io(std::sync::Arc::new(
                std::io::Error::new(std::io::ErrorKind::InvalidData, e),
            ))))
    }

    /// Write rows as XML elements
    fn write_rows<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        rows: &[DataRow],
        element_name: &str,
    ) -> Result<()> {
        for row in rows {
            // Write newline and indent
            writer.write_event(Event::Text(BytesText::new("\n    ")))?;

            // Start element with id attribute
            let mut elem = BytesStart::new(element_name);
            elem.push_attribute(("id", row.id.as_str()));
            writer.write_event(Event::Start(elem))?;

            // Write fields as child elements
            for (key, value) in &row.fields {
                writer.write_event(Event::Text(BytesText::new("\n      ")))?;

                let field_elem = BytesStart::new(key.as_str());
                writer.write_event(Event::Start(field_elem))?;

                let text = match value {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => String::new(),
                    _ => value.to_string(),
                };
                writer.write_event(Event::Text(BytesText::new(&text)))?;

                writer.write_event(Event::End(BytesEnd::new(key.as_str())))?;
            }

            // Close element
            writer.write_event(Event::Text(BytesText::new("\n    ")))?;
            writer.write_event(Event::End(BytesEnd::new(element_name)))?;
        }

        // Final newline before parent close
        writer.write_event(Event::Text(BytesText::new("\n  ")))?;

        Ok(())
    }

    /// Write to a file atomically (write to .tmp, then rename)
    pub fn write_to_file(content: &str, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("xml.tmp");

        // Write to temp file
        fs::write(&tmp_path, content)?;

        // Atomic rename
        fs::rename(&tmp_path, path)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_insert_rows() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <items>
  </items>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());

        let mut fields = HashMap::new();
        fields.insert("name".to_string(), Value::String("test".to_string()));
        fields.insert("value".to_string(), Value::Number(42.into()));

        let rows = vec![DataRow {
            id: "new_1".to_string(),
            fields,
        }];

        let result = modifier.insert_rows(&rows, "items", "item").unwrap();

        assert!(result.contains(r#"<item id="new_1">"#));
        assert!(result.contains("<name>test</name>"));
        assert!(result.contains("<value>42</value>"));
    }

    #[test]
    fn test_insert_empty_parent() {
        let xml = r#"<?xml version="1.0"?>
<root>
  <items/>
</root>"#;

        let modifier = XmlModifier::from_string(xml.to_string());

        let rows = vec![DataRow {
            id: "1".to_string(),
            fields: HashMap::new(),
        }];

        let result = modifier.insert_rows(&rows, "items", "item").unwrap();

        assert!(result.contains(r#"<item id="1">"#));
        assert!(result.contains("</items>"));
    }

    #[test]
    fn test_parent_not_found() {
        let xml = r#"<?xml version="1.0"?><root></root>"#;
        let modifier = XmlModifier::from_string(xml.to_string());

        let result = modifier.insert_rows(&[], "nonexistent", "item");
        assert!(result.is_err());
    }
}
