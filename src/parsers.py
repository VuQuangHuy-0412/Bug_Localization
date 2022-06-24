import glob
import os.path
from collections import OrderedDict
from datetime import datetime
import javalang
import pygments
import csv
from pygments.lexers import JavaLexer
from pygments.token import Token


class BugReport:
    """Class representing each bug report"""

    __slots__ = ['bug_id', 'raw_summary', 'summary', 'raw_description', 'description', 'files', 'report_time',
                 'pos_tagged_summary', 'pos_tagged_description', 'stack_traces']

    def __init__(self, bug_id, raw_summary, summary, raw_description, description, files, report_time):
        self.bug_id = bug_id
        self.raw_summary = raw_summary
        self.summary = summary
        self.raw_description = raw_description
        self.description = description
        self.files = files
        self.report_time = report_time
        self.pos_tagged_summary = None
        self.pos_tagged_description = None
        self.stack_traces = None


class SourceFile:
    """Class representing each source file"""

    __slots__ = ['all_content', 'comments', 'class_names', 'attributes',
                 'method_names', 'variables', 'file_name', 'pos_tagged_comments',
                 'exact_file_name', 'package_name']

    def __init__(self, all_content, comments, class_names, attributes,
                 method_names, variables, file_name, package_name):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None


class Parser:
    """Class containing different parsers"""

    __slots__ = ['name', 'src', 'bug_repo']

    def __init__(self, project):
        self.name = project.name
        self.src = project.src
        self.bug_repo = project.bug_repo

    def report_parser(self):
        """Parse tsv format bug reports"""

        # Convert tsv bug repository to a dictionary
        with open(self.bug_repo, 'r', encoding="ISO-8859-1") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')

            # Iterate through bug reports and build their objects
            bug_reports = OrderedDict()

            for bug_report in reader:
                bug_report['files'] = [os.path.normpath(file)
                                       for file in bug_report['files'].strip().split()
                                       if file.endswith('.java')]
                bug_report['report_time'] = datetime.strptime(bug_report['report_time'], "%Y-%m-%d %H:%M:%S")
                bug_reports[bug_report['bug_id']] = BugReport(bug_report['bug_id'], 
                                                            bug_report['summary'], bug_report['summary'],
                                                            bug_report['description'], bug_report['description'],
                                                            bug_report['files'],
                                                            bug_report['report_time'])

            return bug_reports

    def src_parser(self):
        """Parse source code directory of a program and collect
        its java files.
        """

        # Getting the list of source files recursively from the source directory
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)

        # Creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()

        src_files = OrderedDict()

        # Looping to parse each source file
        for src_file in src_addresses:
            with open(src_file, encoding="ISO-8859-1") as file:
                src = file.read()

            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # Source parsing
            parse_tree = None
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass

            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)

            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    # Removing the license comment
                    if i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])

            # Get the package declaration if exists
            if parse_tree and parse_tree.package:
                package_name = parse_tree.package.name
            else:
                package_name = None

            #if self.name == 'aspectj':
            src_files[os.path.relpath(src_file, start=self.src)] = SourceFile(
                src, comments, class_names, attributes,
                method_names, variables,
                [os.path.basename(src_file).split('.')[0]],
                package_name
            )
            '''else:
                # If source file has package declaration
                if package_name:
                    src_id = (package_name + '.' +
                              os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)

                src_files[src_id] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name
                )'''

        return src_files


def test():
    import datas

    parser = Parser(datas.tomcat)
    br_parser = parser.report_parser()
    src_parser = parser.src_parser()

    src_id, src = list(src_parser.items())[1]
    print(src_id, src.exact_file_name, src.package_name)

    bug_id, bug = list(br_parser.items())[10]
    print(bug_id, bug.files)


if __name__ == '__main__':
    test()