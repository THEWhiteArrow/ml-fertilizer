import os
from pathlib import Path
import re
from typing import List


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run the fertilizer compressor.")
    parser.add_argument("-f", type=str, default="ml_fertilizers/run.py")
    parser.add_argument(
        "-d",
        type=str,
        default=str(Path(os.path.dirname(os.path.abspath(__file__))).parent),
    )
    return vars(parser.parse_args())


parser = get_parser()

file_path = Path(parser["f"])
dir_path = Path(parser["d"])
if not file_path.exists():
    raise FileNotFoundError(f"File {file_path} does not exist.")


print("file_path:", file_path)
print("dir_path:", dir_path)


def dfs(path: Path, already_imported: List[Path] = []) -> List[str]:
    ans: List[str] = []
    lines = open(path, "r").readlines()

    prev_line_import = ""
    prev_line_from = ""
    for line in lines:
        if not line or line.startswith("#"):
            continue

        if line.startswith("import") or prev_line_import != "":
            if "(" in line and prev_line_import == "":
                prev_line_import += line
                continue
            elif prev_line_import != "" and ")" not in line:
                prev_line_import += line
                continue
            elif prev_line_import != "" and ")" in line:
                prev_line_import += line
                line = prev_line_import
                prev_line_import = ""

            ans.append(line)
        elif line.startswith("from") or prev_line_from != "":
            if "(" in line and prev_line_from == "":
                prev_line_from += line
                continue
            elif prev_line_from != "" and ")" not in line:
                prev_line_from += line
                continue
            elif prev_line_from != "" and ")" in line:
                prev_line_from += line
                line = prev_line_from
                prev_line_from = ""

            import_path_name = line.split("from")[1].strip().split("import")[0].strip()
            path_to_import = dir_path / import_path_name.replace(".", "/")

            if path_to_import in already_imported:
                continue

            if path_to_import.with_suffix(".py").exists():
                ans.extend(
                    dfs(
                        path_to_import.with_suffix(".py"),
                        already_imported=already_imported,
                    )
                )
                already_imported.append(path_to_import)
            elif (path_to_import / "__init__.py").exists():
                ans.extend(
                    dfs(
                        path_to_import / "__init__.py",
                        already_imported=already_imported,
                    )
                )
                already_imported.append(path_to_import)
            else:
                ans.append(line)
        elif re.match(r"if __name__ == ['\"]__main__['\"]:", line) and len(ans) > 0:
            break
        else:
            ans.append(line)

    # CLEANING

    ans = [line.replace("__file__", "'file'") for line in ans]
    return ans


final = dfs(file_path)
output_file = file_path.with_suffix(".compressed.py")
with open(output_file, "w") as f:
    for line in final:
        f.write(line)
print(f"Compressed code written to {output_file}")
