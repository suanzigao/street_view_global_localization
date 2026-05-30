FINAL_PREDICTION_PROMPT = """
You have reached the maximum number of navigation steps. Now you must predict your initial GPS coordinates (latitude and longitude) based on ALL the visual information you have gathered.

You have been provided with:
1. The full navigation history and steps taken
2. All Street View images from Step 0 (initial position) through Step {max_steps}
3. The movement rules you were given

Your task is to predict the initial GPS coordinates where you started, and explain your reasoning based on specific visual clues from the images.

**RESPONSE FORMAT:**
Respond only in the following format:

predicted_initial_coords: [LATITUDE],[LONGITUDE] | reasoning: [DETAILED VISUAL CLUES]

For example:
predicted_initial_coords: 40.7580,-73.9855 | reasoning: I saw Times Square signs, New York yellow cabs, and the distinctive architecture of Manhattan.

**IMPORTANT:**
- Your prediction should be as accurate as possible, within roughly 100km of the true location.
- Use specific visual evidence from the images to support your prediction.
- Do not guess wildly; base your prediction on the visual cues you've collected.
- Make sure your reasoning refers to specific things you saw in the images.
"""
