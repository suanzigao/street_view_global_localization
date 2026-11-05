FINAL_PREDICTION_PROMPT = """
You have reached the maximum number of navigation steps. Now you MUST predict your initial GPS coordinates (latitude and longitude) based on ALL the visual information you have gathered.

**YOU MUST ALWAYS PROVIDE GPS COORDINATES - NO EXCEPTIONS!**
- Even if you are unsure, make your BEST ESTIMATE based on visual clues
- NEVER say "I cannot predict", "I can't determine", or any similar refusal
- If you have no clues at all, guess based on the region you think it might be
- ALWAYS format your answer as specified below

========== EVIDENCE PROVIDED ==========

You have been provided with:
1. All the Street View images you captured during navigation (Step 0 = initial position, up to your last step), each labeled with its step number
2. Segmented images with highlighted objects (if segmentation was requested during navigation), labeled with the objects that were segmented
3. Any places found via Google Places API search (shown below, if any)

{{PLACES_HINTS}}

{{HYPOTHESIS}}

{{EVIDENCE_BANK}}

========== PREDICTION STRATEGY ==========

**STEP 1: USE PLACES API RESULTS FIRST (IF AVAILABLE)**
- If you found ANY place via Places API search, use those coordinates as your BASE
- Example: If Places found "Nykredit Bank" at (55.682, 12.570), your prediction should be NEAR this location
- Adjust the coordinates slightly based on visual clues (within 500 meters)

**STEP 2: ANALYZE VISUAL CLUES IF NO PLACES FOUND**
- Look at architectural styles: Nordic (Denmark, Sweden), Mediterranean (Italy, Spain), etc.
- Look at language on signs: Danish, German, French, etc.
- Look at license plates: EU plates (European countries), specific country markers
- Look at vegetation: Tropical (southern), Coniferous (northern), Deciduous (temperate)
- Look at street signs: Stop signs, speed limits, traffic rules
- Look at vehicles: Common car brands, driving side

**STEP 3: ESTIMATE COORDINATES BASED ON REGION**
- Northern Europe: Latitude ~55-70, Longitude ~5-30
- Central Europe: Latitude ~45-55, Longitude ~5-25
- Southern Europe: Latitude ~35-45, Longitude ~-10-30
- Eastern Europe: Latitude ~45-60, Longitude ~20-40
- South America: Latitude ~-5 to ~-35, Longitude ~-80 to ~-35
- North America: Latitude ~25-50, Longitude ~-125 to ~-65
- Asia: Latitude ~10-55, Longitude ~70-145

========== IMPORTANT RULES ==========

**CRITICAL: GOOGLE PLACES API GPS COORDINATES ARE YOUR MOST IMPORTANT EVIDENCE!**
- The coordinates obtained from Places API searches are VERIFIED REAL-WORLD locations
- These GPS coordinates (latitude, longitude) represent the EXACT location of businesses/buildings you identified
- **MANDATORY RULE: If you found ANY place via search, your prediction MUST use coordinates within 1km of that location**
- The coordinates are not estimates - they are GROUND TRUTH from Google's database
- **DO NOT override Places API results with your visual analysis** - visual clues are secondary evidence only
- If Places API found a location at (47.5310, 19.0697), your prediction should be approximately (47.53, 19.07), NOT somewhere else!
- Visual analysis can only help you refine the exact position within 1km, NOT change the city or country!

**HOW TO USE SEGMENTED IMAGES:**
- The highlighted (masked) areas indicate objects you selected as potentially important for location identification
- Analyze the spatial distribution of highlighted objects across different views
- Use the object types and their positions to infer geographic context
- For example: European-style buildings, Asian script on signs, tropical vegetation, etc.

========== RESPONSE FORMAT ==========

Respond ONLY in the following format:

predicted_initial_coords: [LATITUDE],[LONGITUDE] | reasoning: [DETAILED VISUAL CLUES]

**CRITICAL: PRECISION REQUIREMENT - ALWAYS use 6 decimal places for GPS coordinates!**
- GPS coordinates MUST be formatted as: LATITUDE,LONGITUDE with EXACTLY 6 decimal places
- WRONG: "55.682,12.570" or "55.6820,12.5700" (not 6 decimal places!)
- CORRECT: "55.682000,12.570000" (exactly 6 decimal places)
- WRONG: "-23.55,-46.63" (only 2 decimal places!)
- CORRECT: "-23.550000,-46.630000" (exactly 6 decimal places)

**EXAMPLE 1 (with Places API):**
predicted_initial_coords: 55.682000,12.570000 | reasoning: I saw "Nykredit" on a building. Places API confirmed this is Copenhagen, Denmark at (55.682000, 12.570000).

**EXAMPLE 2 (without Places API):**
predicted_initial_coords: 52.520000,13.405000 | reasoning: The architecture looks German with brick buildings and street signs. The language appears to be German. This suggests Berlin, Germany around (52.520000, 13.405000).

**EXAMPLE 3 (uncertain - must still provide coordinates):**
predicted_initial_coords: 48.856000,2.352000 | reasoning: The buildings look French with balconies and wrought iron. The vegetation is deciduous, suggesting a temperate climate. I'll guess Paris, France around (48.856000, 2.352000).

**EXAMPLE 4 (South America):**
predicted_initial_coords: -23.554800,-46.624800 | reasoning: The architecture shows high-rise buildings typical of Brazil. The vegetation includes palm trees, suggesting a tropical climate. Portuguese language on signs confirms Brazil. This suggests São Paulo around (-23.554800, -46.624800).

**EXAMPLE 5 (Eastern Europe):**
predicted_initial_coords: 55.646400,37.725000 | reasoning: The architecture and street signs suggest Russia or Eastern Europe. The buildings have a distinctive Soviet-era style. This suggests Moscow around (55.646400, 37.725000).

**IMPORTANT:**
- GPS coordinates MUST have exactly 6 decimal places - this is MANDATORY
- If Places API results are available, your prediction MUST be within 1km of those coordinates
- Use Places API GPS as GROUND TRUTH - this is the most reliable information you have
- Use BOTH original images AND segmented images for your analysis
- Pay special attention to the highlighted objects in segmented images - they indicate what you found most relevant during navigation
- Use specific visual evidence from the images to support your prediction
- Do not guess wildly; base your prediction on the visual cues you've collected
- Make sure your reasoning refers to specific things you saw in the images, including segmented objects
- Always provide latitude first, then longitude, separated by a comma
- **FAILURE TO COMPLY:** Any response that does NOT contain valid GPS coordinates with 6 decimal places will be treated as a FAILURE
"""
