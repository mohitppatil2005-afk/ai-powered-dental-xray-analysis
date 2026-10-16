
ANALYSIS_MODULES = {
    "Tooth segmentation": {
        "configured": False,
        "description": "Identify individual teeth in the X-ray."
    },
    "Dental caries": {
        "configured": False,
        "description": "Highlight possible cavity regions."
    },
    "Periodontal bone loss": {
        "configured": False,
        "description": "Assess possible periodontal bone loss."
    },
    "Periapical abnormalities": {
        "configured": False,
        "description": "Highlight possible abnormalities near tooth roots."
    },
}


def get_module_status():
    return {
        name: {
            **details,
            "status": (
                "Model configured"
                if details["configured"]
                else "Model not configured — no prediction generated"
            )
        }
        for name, details in ANALYSIS_MODULES.items()
    }
