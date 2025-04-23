import xml.etree.ElementTree as ET
import pandas as pd
import argparse as ap
import os

"""
Tag explanation:
au = author
p = paragraph
lg = image caption
a = passage with a hyperlink
ld = lead
tx = whole text 
zt = subtitle
"""

def get_lead(content):
    try:
        root = ET.fromstring(content)
    except:
        return []
    lead = root.find('ld')
    if lead is None:
        return []
    paras = [para.text for para in lead.findall('p')]
    paras = [para for para in paras if para]
    assert len(paras) == 1
    return paras

def get_paragraph_list(content):
    try:
        root = ET.fromstring(content)
    except:
        return []
    paras = [get_full_text(para).replace('\xa0', '') for para in root.findall('.//p')]
    paras = [para for para in paras if para]
    try:
        assert len(paras) > 0
    except:
        print(f"Error parsing content: {content}")
        print(f"Error parsing paras: {paras}")
        return []
    return paras

def get_subtitles(content):
    root = ET.fromstring(content)
    subtitles = [zt.text for zt in root.iter('zt')]
    subtitles = [subtitle for subtitle in subtitles if subtitle]
    assert len(subtitles) > 0
    return subtitles
    
def get_full_text(element):
    """Recursively get the text of an element, including children."""
    text = element.text or  ""
    for child in element:
        if child.tag == 'a':  
            text += child.text or ""
        text += child.tail or "" 
    return text

def get_xml_tags(content):
    root = ET.fromstring(content)
    unique_tags = set()
    def traverse_tree(element):
        unique_tags.add(element.tag)
        for child in element:
            traverse_tree(child)
    traverse_tree(root)
    return unique_tags


def main():

    parser = ap.ArgumentParser()
    parser.add_argument("file", type=str, help="path to the file")
    parser.add_argument("--info", type=str, help="specify what to parse. Options: lead, paragraphs, subtitles, full_text, unique_tags")
    parser.add_argument("--output-type", type=str, help="output type. Options: stdout (print to console), file (write to file)")
    args = parser.parse_args()

    df = pd.read_csv(args.file, sep='\t', encoding='utf-8')

    if args.output_type == "stdout":
        pass
    elif args.output_type == "file":
        os.makedirs('parsed_texts', exist_ok=True)
        f = open('parsed_texts/' + args.file.split('.')[0].split('/')[-1] + '_parsed_' + args.info + '.txt', 'w')


    if args.info == "lead":
        for index, row in df.iterrows():
            content = row['content']
            print(content)
            leads = get_lead(content)
            print(leads)
            if args.output_type == "file":
                if type(leads) == list:
                    for _ in leads:
                        f.write(_+"\n")
                else:
                    f.write(leads+"\n")
    elif args.info == "paragraphs":
        for index, row in df.iterrows():
            content = row['content']
            paragraphs = get_paragraph_list(content)
            print(paragraphs)
            if args.output_type == "file":
                if type(paragraphs) == list:
                    for _ in paragraphs:
                        f.write(_+"\n")
                else:
                    f.write(paragraphs+"\n")
    elif args.info == "unique_tags":
        for index, row in df.iterrows():
            content = row['content']
            unique_tags = get_xml_tags(content)
            print(unique_tags)
            if args.output_type == "file":
                for _ in unique_tags:
                    f.write(_+"\n")
    
    elif args.info == "aligned_titles" and 'alig' in args.file:
        print("Aligning titles")
        pair_id_prev = None
        prev_title = None
        for index, row in df.iterrows():
            if pair_id_prev:
                if row['pair_id'] == pair_id_prev:
                    print(row['pair_id'], prev_title, row['head'])
                    f.write(prev_title + "\t" + row['head'] + "\n")
                else:
                    prev_title = row['head']
                    pair_id_prev = row['pair_id']
            else:
                prev_title = row['head']
                pair_id_prev = row['pair_id']

    elif args.info == "aligned_leads" and 'align' in args.file:
        print("Aligning leads")
        pair_id_prev = None
        prev_lead = None
        for index, row in df.iterrows():
            if pair_id_prev:
                if row['pair_id'] == pair_id_prev:
                    print(row['pair_id'], prev_lead, " ".join(get_lead(row['content'])))
                    f.write(prev_lead + "\t" + " ".join(get_lead(row['content'])) + "\n")
                else:
                    prev_lead = " ".join(get_lead(row['content']))
                    pair_id_prev = row['pair_id']
            else:
                prev_lead = " ".join(get_lead(row['content']))
                pair_id_prev = row['pair_id']

    if args.output_type == "file":
        f.close()

if __name__ == "__main__":
    main()