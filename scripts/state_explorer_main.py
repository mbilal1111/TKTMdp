import sys
from collections import OrderedDict

def extract_solar_power_ids(file_path):
    """
    Parse the file as text and extract active power and flex power IDs for HEAT-PUMPs.
    (This version fixes the “end_index” so that each solar_panel_ids entry is just "HEAT-PUMP:<UUID>",
    then groups by unique panel to show each only once and prints tidy “easy copy” lists.)
    """
    try:
        # Read the entire file as text
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Temporary arrays to store raw results (one entry per occurrence in the file)
    solar_panel_ids = []
    active_power_ids = []
    flex_power_ids = []
    
    # Split by the HEAT-PUMP: identifier to find all HEAT-PUMPs
    parts = content.split("HEAT-PUMP:")
    
    # Skip the first part (before the first occurrence)
    for part in parts[1:]:
        # ─────────────────────────────────────────────────────────────
        # Replace the old "find('","ref":')" with a simple find('"')
        # so that we stop right at the first double‐quote after the ID.
        end_index = part.find('"')
        # ─────────────────────────────────────────────────────────────
        
        if end_index != -1:
            # Extract exactly "HEAT-PUMP:<uuid>"
            panel_id = "HEAT-PUMP:" + part[:end_index]
            solar_panel_ids.append(panel_id)
            
            # Find all dynamicRefs for this panel
            dynamic_start = part.find('dynamicRefs":')
            dynamic_end = part.find(']', dynamic_start)
            if dynamic_start != -1 and dynamic_end != -1:
                dynamic_section = part[dynamic_start:dynamic_end]
                
                # Look for ACTIVE-POWER-3P within this section
                active_power_start = dynamic_section.find('ACTIVE-POWER-3P:')
                if active_power_start != -1:
                    active_power_end = dynamic_section.find('"', active_power_start)
                    if active_power_end != -1:
                        raw_id = dynamic_section[active_power_start:active_power_end]
                        active_id = raw_id.replace(":", "_")  
                        active_power_ids.append(active_id)
                    else:
                        active_power_ids.append(None)
                else:
                    active_power_ids.append(None)
                
                # Look for FLEX-POWER within this section
                flex_power_start = dynamic_section.find('FLEX-POWER:')
                if flex_power_start != -1:
                    flex_power_end = dynamic_section.find('"', flex_power_start)
                    if flex_power_end != -1:
                        raw_id = dynamic_section[flex_power_start:flex_power_end]
                        flex_id = raw_id.replace(":", "_")
                        flex_power_ids.append(flex_id)
                    else:
                        flex_power_ids.append(None)
                else:
                    flex_power_ids.append(None)
            else:
                active_power_ids.append(None)
                flex_power_ids.append(None)
    
    # ────────────────────────────────────────────────────────────────────────
    # Group by each unique panel ID, keeping the first non-None active/flex IDs
    # ────────────────────────────────────────────────────────────────────────
    unique_panels = OrderedDict()
    for pid, a_id, f_id in zip(solar_panel_ids, active_power_ids, flex_power_ids):
        if pid not in unique_panels:
            # Initialize entry with whatever IDs we see (may be None)
            unique_panels[pid] = {"active_id": a_id, "flex_id": f_id}
        else:
            # If we later encounter a non-None ID, overwrite the stored value
            if a_id:
                unique_panels[pid]["active_id"] = a_id
            if f_id:
                unique_panels[pid]["flex_id"] = f_id
    
    # ────────────────────────────────────────────────────────────────────────
    # Print the cleaned‐up, one‐entry‐per‐panel results
    # ────────────────────────────────────────────────────────────────────────
    print("\n===== HEAT-PUMPS POWER IDs =====")
    if unique_panels:
        print(f"Found {len(unique_panels)} unique HEAT-PUMPs\n")
        for idx, (panel_uuid, ids) in enumerate(unique_panels.items(), start=1):
            active_id = ids["active_id"] or "None"
            flex_id   = ids["flex_id"]   or "None"
            print(f"{idx}. HEAT-PUMP: {panel_uuid}")
            print(f"   Active Power ID: {active_id}")
            print(f"   Flex Power ID:   {flex_id}\n")
        
        # ────────────────────────────────────────────────────────────────────
        # Build “easy‐copy” lists for all non‐None IDs, deduplicated in original order
        # ────────────────────────────────────────────────────────────────────
        all_active_ids = []
        all_flex_ids   = []
        for ids in unique_panels.values():
            if ids["active_id"]:
                all_active_ids.append(ids["active_id"])
            if ids["flex_id"]:
                all_flex_ids.append(ids["flex_id"])
        
        def dedupe_keep_order(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        
        all_active_ids = dedupe_keep_order(all_active_ids)
        all_flex_ids   = dedupe_keep_order(all_flex_ids)
         
    else:
        print("No HEAT-PUMPs found.")
    
    return solar_panel_ids, active_power_ids, flex_power_ids


if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to the file: ")
    
    extract_solar_power_ids(file_path)
