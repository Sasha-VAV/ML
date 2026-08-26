from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

SKILL_RESOURCES: dict[str, tuple[str, ...]] = {
    "destination-advisor": ("places.md",),
    "trip-booking": (),
}


class Skill:
    def __init__(self, name: str, version: str, description: str, system_prompt: str):
        self.name = name
        self.version = version
        self.description = description
        self.system_prompt = system_prompt


def load_skill(dir_name: str, resource_files: tuple[str, ...] = ()) -> Skill:
    skill_dir = SKILLS_DIR / dir_name
    raw = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    _, frontmatter_raw, body = raw.split("---", 2)
    frontmatter = yaml.safe_load(frontmatter_raw)

    parts = [body.strip()]
    for resource_file in resource_files:
        parts.append(f"# {resource_file}\n" + (skill_dir / resource_file).read_text(encoding="utf-8").strip())

    return Skill(
        name=frontmatter["name"],
        version=frontmatter["version"],
        description=" ".join(frontmatter["description"].split()),
        system_prompt="\n\n".join(parts),
    )


def load_all_skills() -> dict[str, Skill]:
    return {
        name: load_skill(name, resource_files=resources)
        for name, resources in SKILL_RESOURCES.items()
    }
