import unittest
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from parse_article import get_lead, get_paragraph_list, get_subtitles
import pandas as pd

class TestTSVFileStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class variables for the file path and report file from command-line arguments."""
        parser = argparse.ArgumentParser(description="TSV file structure tester.")
        parser.add_argument("tsv_file", help="Path to the TSV file to be tested.")
        args = parser.parse_args()

        cls.FILE_PATH = args.tsv_file
        cls.REPORT_FILE = "missing_cells_report.txt"

        if not os.path.exists(cls.FILE_PATH):
            print(f"Error: The file '{cls.FILE_PATH}' does not exist.")
            sys.exit(1)

    def test_file_exists(self):
        """Check if the TSV file exists."""
        self.assertTrue(os.path.exists(self.FILE_PATH), "TSV file does not exist.")

    def test_file_not_empty(self):
        """Ensure the TSV file is not empty."""
        self.assertGreater(os.path.getsize(self.FILE_PATH), 0, "TSV file is empty.")

    def test_valid_tsv_structure(self):
        """Check that all rows have the correct number of columns."""
        with open(self.FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertGreater(len(lines), 1, "TSV file should have at least one header row and one data row.")

        header = lines[0].strip().split("\t")
        expected_columns = len(header)

        self.assertGreater(expected_columns, 1, "TSV file should have at least two columns.")
        self.assertTrue(all(header), "Header row contains empty column names.")

        for i, line in enumerate(lines[1:], start=2):  # Skip header, count from line 2
            row = line.strip().split("\t")
            # print which column is missing
            if len(row) < expected_columns:
                print(f"Row {i} has {len(row)} columns, expected {expected_columns}.")
                # print which column is missing
                for j in range(expected_columns):
                    if j >= len(row):
                        print(f"Column {j} is missing")

            self.assertEqual(len(row), expected_columns, f"Row {i} has {len(row)} columns, expected {expected_columns}.")


    def test_no_empty_rows(self):
        """Check that there are no completely empty rows."""
        with open(self.FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertTrue(all(line.strip() for line in lines), "File contains empty rows.")

    def test_no_empty_cells(self):
        """Check for empty cells in any row and generate a report if found."""
        missing_cells = []

        with open(self.FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header = lines[0].strip().split("\t")
        for i, line in enumerate(lines[1:], start=2):  # Start from line 2 (excluding header)
            row = line.strip().split("\t")
            for j, cell in enumerate(row):
                if cell.strip() == "":
                    missing_cells.append(f"Empty cell found at row {i}, column '{header[j]}'")

        if missing_cells:
            with open(self.REPORT_FILE, "w", encoding="utf-8") as report:
                report.write("Missing Cells Report:\n")
                report.write("\n".join(missing_cells))
                report.write("\n")

            self.fail(f"Empty cells found. See '{self.REPORT_FILE}' for details.")

class TestXMLContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class variables for the file path from command-line arguments."""
        parser = argparse.ArgumentParser(description="TSV file content tester.")
        parser.add_argument("tsv_file", help="Path to the TSV file to be tested.")
        args = parser.parse_args()

        cls.FILE_PATH = args.tsv_file

    def test_xml_parsability(self):
        """Check if only the content column contains XML and can be parsed by parse_article functions."""
        df = pd.read_csv(self.FILE_PATH, sep='\t')

        header = df.columns.to_list()

        try:
            content_idx = header.index("content")
        except ValueError:
            self.fail("No 'content' column found in TSV file")

        parsing_errors = []
        for i, row in df.iterrows():
            # iterate over values in row
            for j, cell in enumerate(row):
                # check that non-content columns don't contain XML tags
                if j != content_idx and isinstance(cell, str) and ("<" in cell or ">" in cell):
                    parsing_errors.append(f"Row {i}: Column '{header[j]}' contains XML tags but should not")

            # Test content column XML
            content = row['content']
                
            # Test if content can be parsed by parse_article functions
            try:
                get_lead(content)
            except (AssertionError, ET.ParseError) as e:
                parsing_errors.append(f"Row {i}: Failed to parse lead - {str(e)}")
            
            try:
                get_paragraph_list(content)
            except (AssertionError, ET.ParseError) as e:
                parsing_errors.append(f"Row {i}: Failed to parse paragraphs - {str(e)}")
            

        if parsing_errors:
            with open("xml_parsing_errors.txt", "w", encoding="utf-8") as report:
                report.write("XML Parsing Errors Report:\n")
                report.write("\n".join(parsing_errors))
                report.write("\n")
            
            self.fail(f"XML parsing errors found. See 'xml_parsing_errors.txt' for details.")

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])  # Avoid passing argparse arguments to unittest