def analyze_change(old_class, new_class):
    """
    Compare classification classes of old and new images and determine change impact.
    
    Rules:
    - Forest -> Residential = Deforestation
    - Forest -> Industrial = Industrial Expansion
    - Crop -> Industrial = Agricultural Land Loss
    - River -> Industrial = Water Resource Risk
    - No change = Stable Land Use
    """
    change_detected = old_class != new_class
    
    if not change_detected:
        return {
            "old_class": old_class,
            "new_class": new_class,
            "change_detected": False,
            "impact": "Stable Land Use"
        }
    
    # Specific rules matching
    if old_class == "Forest" and new_class == "Residential":
        impact = "Deforestation (Urban Expansion)"
    elif old_class == "Forest" and new_class == "Industrial":
        impact = "Industrial Expansion (Forest Loss)"
    elif old_class in ["AnnualCrop", "PermanentCrop"] and new_class == "Industrial":
        impact = "Agricultural Land Loss (Industrial Development)"
    elif old_class in ["AnnualCrop", "PermanentCrop"] and new_class == "Residential":
        impact = "Agricultural Land Loss (Urban Expansion)"
    elif old_class == "River" and new_class == "Industrial":
        impact = "Water Resource Risk (Industrial Runoff / Dev)"
    elif new_class in ["Residential", "Industrial", "Highway"]:
        impact = f"Urban Expansion & Infrastructure Growth ({old_class} to {new_class})"
    elif new_class == "Forest" and old_class not in ["Forest", "SeaLake"]:
        impact = "Afforestation / Natural Regeneration"
    elif new_class in ["SeaLake", "River"] and old_class not in ["SeaLake", "River"]:
        impact = "Inundation / Water Body Expansion"
    else:
        impact = f"Land Cover Transition ({old_class} to {new_class})"
        
    return {
        "old_class": old_class,
        "new_class": new_class,
        "change_detected": True,
        "impact": impact
    }
