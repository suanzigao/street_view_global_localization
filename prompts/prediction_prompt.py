FINAL_PREDICTION_PROMPT = """
Your task is to predict the initial GPS coordinates where you started, and explain your reasoning based on specific visual clues from the images.

**RESPONSE FORMAT:**
Respond only in the following format:

predicted_initial_coords: [LATITUDE],[LONGITUDE] | reasoning: [DETAILED VISUAL CLUES]

For example:
predicted_initial_coords: 40.7580,-73.9855 | reasoning: I saw Times Square signs, New York yellow cabs, and the distinctive architecture of Manhattan.

**IMPORTANT:**
- Do not guess wildly; base your prediction on the visual cues you've collected.
- Make sure your reasoning refers to specific things you saw in the images.
"""
