import re
import json
import math
import hashlib
import textwrap
import itertools
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from importlib.resources import files
from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from graphviz import Digraph

#################################
##### Parse Raw ISU Catalog #####
#################################

def fParseISURawCatalog(raw_txt: Path) -> List[str]:
    with open(raw_txt, 'r', encoding='utf-8') as f:
        text = f.read()

    course_pattern = r"(?:^|\n)([A-Z]{2,4} \d{4}[A-Z]?: .+?)(?=\n[A-Z]{2,4} \d{4}[A-Z]?: |\Z)"

    matches = re.findall(course_pattern, text, re.DOTALL)
    
    return [course.strip() for course in matches]

def fParseISUCourseDesc(description: str, client: OpenAI):
    prompt = f"""
You are a helpful assistant that parses university course descriptions. Given a raw course description below, output a structured JSON object with the following fields:

- code: string (e.g., "CSCI 2021")
- name: string
- credits: integer
- prerequisites: a list of course codes (e.g., ["MATH 1650", ["MATH 1630X", "MATH 1640X"]]). Use nested lists only for OR conditions. Make sure to use brackets.
- corequisites: same format as prerequisites
- typically_offered: a list of terms, like ["Fall", "Spring", "Summer"]

Ignore non-course prerequisites like GPA or honors membership. Ignore restrictions or grading basis.

Return only the JSON and nothing else so your output can be parsed as a json file. Here is an example of the expected format:
{{
    "code": "CSCI 2021",
    "name": "Data Structures and Algorithms",
    "credits": 3,
    "prerequisites": ["CSCI 1410", ["MATH 1630X", "MATH 1640X"]],
    "corequisites": ["CSCI 2021L"],
    "typically_offered": ["Fall", "Spring"]
}}

Now parse this course:
----------------------

{description}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Extract and clean JSON
    reply = response.choices[0].message.content

    #parse only the JSON part
    json_start = reply.find('{')
    json_end = reply.rfind('}') + 1
    if json_start == -1 or json_end == -1:
        print("No JSON object found in the response. Raw output:")
        print(reply)
        return None
    
    reply = reply[json_start:json_end]

    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:")
        print(reply)
        return None

def fBuildISUCourseCatalogCSV(class_folder: Path, csv_catalog: Path, raw_catalog: Path, client):
    class_folder.mkdir(parents=True, exist_ok=True)
    csv_catalog.parent.mkdir(parents=True, exist_ok=True)

    class_desc = fParseISURawCatalog(raw_catalog)

    for c in tqdm(class_desc, desc="Parsing course descriptions"):
        code_match = re.match(r"([A-Z]{2,4} \d{4}[A-Z]?):", c)
        
        if code_match is None:
            continue
        
        code = code_match.group(1)
        if (class_folder / f"{code}.json").exists():
            continue
        
        dict = fParseISUCourseDesc(c, client)

        if dict is None:
            continue

        if "code" not in dict.keys():
            continue

        code = dict['code']
        with open(class_folder / f"{code}.json", 'w', encoding='utf-8') as f:
            json.dump(dict, f, indent=4)
    
        # from the folder of json files, write to a csv file with columns code, name, credits, prerequisites, corequisites, typically_offered
    with open(csv_catalog, 'w', encoding='utf-8') as f:
        f.write("code,name,credits,prerequisites,corequisites,typically_offered\n")
        for json_file in tqdm(class_folder.glob("*.json"), desc="Writing CSV catalog"):
            with open(json_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                code = data.get("code", "")
                name = data.get("name", "").replace(",", ";")
                credits = data.get("credits", "")
                prerequisites = json.dumps(data.get("prerequisites", [])).replace('"', '""')
                corequisites = json.dumps(data.get("corequisites", [])).replace('"', '""')
                typically_offered = json.dumps(data.get("typically_offered", [])).replace('"', '""')

                f.write(f'{code},{name},{credits},"{prerequisites}","{corequisites}","{typically_offered}"\n')


def fLoadISUCourseCatalogCSV(csv_catalog: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_catalog, 
        quotechar='"',
        delimiter=',',
        encoding='utf-8'
    )
    
    return df

#################################
##### ISU Catalog Structure #####
#################################

class ISUTerm(Enum):
    FALL = "Fall"
    SPRING = "Spring"
    SUMMER = "Summer"

    @classmethod
    def valid_str(cls, term: str) -> bool:
        return term.capitalize() in cls._value2member_map_

    @classmethod
    def from_str(cls, term: str) -> "ISUTerm":
        return cls(cls._value2member_map_[term.capitalize()])

    def __repr__(self) -> str:
        return f"{self.name}"

@dataclass
class ISUCourseCode:
    dept: str
    number: int
    modifier: str

    @classmethod
    def valid_str(cls, code: str) -> bool:
        match = re.match(r"([A-Z]{2,4})\s+(\d{4})([A-Z]*)$", code.strip())
        return match

    @classmethod
    def from_str(cls, code: str) -> "ISUCourseCode":
        match = re.match(r"([A-Z]{2,4})\s+(\d{4})([A-Z]*)$", code.strip())
        if not match:
            raise ValueError(f"Invalid course code format: {code}")
        dept, number, modifier = match.groups()
        return cls(dept, int(number), modifier)

    def __repr__(self) -> str:
        mod = self.modifier if self.modifier else ""
        return f"{self.dept} {self.number}{mod}"

ISUPrerequisite = Union[ISUCourseCode, List[ISUCourseCode]]

@dataclass
class ISUCourse:
    code: ISUCourseCode
    name: str
    credits: int
    prerequisites: List[ISUPrerequisite] = field(default_factory=list)
    corequisites: List[ISUPrerequisite] = field(default_factory=list)
    typically_offered: List[ISUTerm] = field(default_factory=list)

ISUCatalog = List[ISUCourse]

#################################
##### Parse ISU CSV Catalog #####
#################################

_NestedStrList = List[Union[str, List[str]]]

def _FilterAndParse_CourseCodes(nested: _NestedStrList) -> List[ISUPrerequisite]:
    if isinstance(nested, str):
        if ISUCourseCode.valid_str(nested):
            return ISUCourseCode.from_str(nested)
        else:
            return None  # Invalid string, remove it
    elif isinstance(nested, list):
        result = []
        for item in nested:
            parsed = _FilterAndParse_CourseCodes(item)
            if parsed is not None:
                result.append(parsed)
        if not result:
            return None
        if len(result) == 1:
            return result[0]
        return result
    else:
        return None  # Unexpected type, skip


def _FilterAndParse_Terms(terms: List[str]) -> List[ISUTerm]:
    result = [ISUTerm.from_str(term) for term in terms if ISUTerm.valid_str(term)]
    return result

def _SafeCSVLoadJsonRow(s, default="[]"):
    if s is None:
        s = default
    elif isinstance(s, float) and math.isnan(s):
        s = default
    return json.loads(s)

def fBuildISUCourseCatalog(csv_catalog: pd.DataFrame) -> List[ISUCourse]:
    courses = []
    for _, row in csv_catalog.iterrows():
        course = ISUCourse(
            ISUCourseCode.from_str(row["code"]),
            row["name"],
            int(row["credits"]),
            _FilterAndParse_CourseCodes(_SafeCSVLoadJsonRow(row["prerequisites"])),
            _FilterAndParse_CourseCodes(_SafeCSVLoadJsonRow(row["corequisites"])),
            _FilterAndParse_Terms(_SafeCSVLoadJsonRow(row["typically_offered"]))
        )


        course.prerequisites = [] if course.prerequisites is None else course.prerequisites
        course.corequisites = [] if course.corequisites is None else course.corequisites
        course.typically_offered = [] if course.typically_offered is None else course.typically_offered
        
        course.prerequisites = course.prerequisites if isinstance(course.prerequisites, list) else [course.prerequisites]
        course.corequisites = course.corequisites if isinstance(course.corequisites, list) else [course.corequisites]

        courses.append(course)
    
    return courses

#################################
##### Visualize ISU Catalog #####
#################################

def _PrereqToStr(prereq: ISUPrerequisite) -> str:
    if isinstance(prereq, List):
        return "(" + " or ".join([str(p) for p in prereq]) + ")"
    return str(prereq)

def _PrereqListToStr(prereqs: List[ISUPrerequisite]) -> str:
    return ", ".join(_PrereqToStr(p) for p in prereqs) if prereqs else "-"

def _TermsToStr(terms: List[ISUTerm]) -> str:
    return ", ".join(term.value for term in terms) if terms else "-"

def fISUCatalogToStr(catalog: List[ISUCourse], cell_width_limit=40) -> str:
    catalog = sorted(catalog, key=lambda c: (c.code.dept, c.code.number, c.code.modifier))
    
    col_widths = {
        "Code": min(max(len(str(course.code)) for course in catalog), cell_width_limit) if catalog else 4,
        "Name": min(max(len(course.name) for course in catalog), cell_width_limit) if catalog else 4,
        "Credits": 7,
        "Prerequisites": min(max(len(_PrereqListToStr(course.prerequisites)), cell_width_limit) for course in catalog) if catalog else 12,
        "Corequisites": min(max(len(_PrereqListToStr(course.corequisites)), cell_width_limit) for course in catalog) if catalog else 11,
        "Offered": min(max(len(_TermsToStr(course.typically_offered)), cell_width_limit) for course in catalog) if catalog else 7,
    }

    header = (
        f"{'Code':<{col_widths['Code']}}  "
        f"{'Name':<{col_widths['Name']}}  "
        f"{'Credits':<{col_widths['Credits']}}  "
        f"{'Prerequisites':<{col_widths['Prerequisites']}}  "
        f"{'Corequisites':<{col_widths['Corequisites']}}  "
        f"{'Offered':<{col_widths['Offered']}}"
    )

    lines = [header]
    lines.append("-" * len(header))

    def wrap_cell(text: str, width: int) -> List[str]:
        # textwrap.wrap respects word boundaries
        return textwrap.wrap(text, width=width) or [""]

    for course in catalog:
        # Prepare wrapped lines for each column cell
        code_lines = wrap_cell(str(course.code), col_widths["Code"])
        name_lines = wrap_cell(course.name, col_widths["Name"])
        credits_lines = wrap_cell(str(course.credits), col_widths["Credits"])
        prereq_lines = wrap_cell(_PrereqListToStr(course.prerequisites), col_widths["Prerequisites"])
        coreq_lines = wrap_cell(_PrereqListToStr(course.corequisites), col_widths["Corequisites"])
        offered_lines = wrap_cell(_TermsToStr(course.typically_offered), col_widths["Offered"])

        # Calculate how many lines this row needs (max of all)
        max_lines = max(
            len(code_lines),
            len(name_lines),
            len(credits_lines),
            len(prereq_lines),
            len(coreq_lines),
            len(offered_lines),
        )

        # Pad shorter columns with empty strings for line-by-line printing
        code_lines += [""] * (max_lines - len(code_lines))
        name_lines += [""] * (max_lines - len(name_lines))
        credits_lines += [""] * (max_lines - len(credits_lines))
        prereq_lines += [""] * (max_lines - len(prereq_lines))
        coreq_lines += [""] * (max_lines - len(coreq_lines))
        offered_lines += [""] * (max_lines - len(offered_lines))

        # Print the row line by line
        for i in range(max_lines):
            lines.append(
                f"{code_lines[i]:<{col_widths['Code']}}  "
                f"{name_lines[i]:<{col_widths['Name']}}  "
                f"{credits_lines[i]:<{col_widths['Credits']}}  "
                f"{prereq_lines[i]:<{col_widths['Prerequisites']}}  "
                f"{coreq_lines[i]:<{col_widths['Corequisites']}}  "
                f"{offered_lines[i]:<{col_widths['Offered']}}"
            )

    return "\n".join(lines)

def fCatalogToGraph(catalog: ISUCatalog) -> Digraph:
    dot = Digraph(comment="ISU Course Prerequisite Graph", format='png')
    dot.attr('node', shape='box')

    departments = sorted(set(course.code.dept for course in catalog))
    color_palette = [
        "#AEDFF7", "#FFD580", "#C3FDB8", "#FFB6C1", "#D9D9D9",
        "#E6E6FA", "#F7CAC9", "#B0E0E6", "#F0E68C", "#D5A6BD"
    ]
    dept_color_map = {dept: color for dept, color in zip(departments, itertools.cycle(color_palette))}

    # Add all course nodes
    for course in catalog:
        course_id = str(course.code)
        dept = course.code.dept
        color = dept_color_map[dept]
        label = f"{course.code}\\n{course.name if len(course.name) < 30 else course.name[:27] + '...'}"
        dot.node(course_id, label=label, fillcolor=color, style="filled", fontname="Arial", fontsize="10")

    # Add edges
    for course in catalog:
        course_id = str(course.code)

        for prereq in course.prerequisites:
            if isinstance(prereq, ISUCourseCode):
                # Single prerequisite
                dot.edge(str(prereq), course_id, style="solid")
            elif isinstance(prereq, list):
                # OR prerequisites
                # Create a dummy node ID based on hash
                hash_id = hashlib.md5(("OR" + course_id + "".join(map(str, prereq))).encode()).hexdigest()[:8]
                or_node_id = f"OR_{hash_id}"
                dot.node(or_node_id, "OR", shape="circle", style="dashed")
                for alt in prereq:
                    dot.edge(str(alt), or_node_id, style="solid")
                dot.edge(or_node_id, course_id, style="solid")

        for coreq in course.corequisites:
            if isinstance(coreq, ISUCourseCode):
                dot.edge(str(coreq), course_id, style="dashed")
            elif isinstance(coreq, list):
                # OR prerequisites
                # Create a dummy node ID based on hash
                hash_id = hashlib.md5(("OR" + course_id + "".join(map(str, coreq))).encode()).hexdigest()[:8]
                or_node_id = f"OR_{hash_id}"
                dot.node(or_node_id, "OR", shape="circle", style="dashed")
                for alt in coreq:
                    dot.edge(str(alt), or_node_id, style="dashed")
                dot.edge(or_node_id, course_id, style="dashed")

    dot.engine = "sfdp"
    dot.graph_attr["overlap"] = "false"
    dot.graph_attr["sep"] = "+30"
    dot.graph_attr["splines"] = "true"
    dot.graph_attr["outputorder"] = "edgesfirst"
    dot.graph_attr["pack"] = "true"
    dot.graph_attr["concentrate"] = "true"

    return dot

