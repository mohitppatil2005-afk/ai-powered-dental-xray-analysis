
import json
from datetime import datetime, timezone


def create_report(filename, module_status):
    """Generate a report without inventing clinical predictions."""
    return {
        "project": "AI-Powered Dental X-Ray Analysis",
        "image_filename": filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Educational and research use only",
        "clinical_warning": (
            "This application is not a diagnostic tool. "
            "A qualified dentist must interpret the X-ray."
        ),
        "modules": module_status,
    }


def report_as_json(report):
    """Convert the report into JSON format."""
    return json.dumps(report, indent=2, ensure_ascii=False)


def report_as_text(report):
    """Convert the report into readable text."""
    lines = [
        report["project"],
        "=" * 45,
        f"Image: {report['image_filename']}",
        f"Generated: {report['generated_at']}",
        "",
        "ANALYSIS MODULES",
        "-" * 45,
    ]

    for name, details in report["modules"].items():
        lines.append(f"{name}: {details['status']}")

    lines.extend([
        "",
        "IMPORTANT",
        report["clinical_warning"],
    ])

    return "\n".join(lines)
