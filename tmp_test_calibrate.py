from app import map_simple_inputs_to_features, _calibrate_predictions
import pandas as pd

good = {'age':'30','gender':'Female','smoking_habit':'non_smoker','alcohol_consumption':'never','exercise_level':'high','water_intake':'high','sleep_hours':'8','fast_food':'rarely','stress_level':'low','pollution_exposure':'low','healthcare_access':'good','healthy_food':'good'}
bad = {'age':'55','gender':'Male','smoking_habit':'regular_smoker','alcohol_consumption':'frequently','exercise_level':'low','water_intake':'low','sleep_hours':'4','fast_food':'daily','stress_level':'high','pollution_exposure':'high','healthcare_access':'poor','healthy_food':'poor'}

fg = pd.DataFrame([map_simple_inputs_to_features(good)])
fb = pd.DataFrame([map_simple_inputs_to_features(bad)])

baseline_death, baseline_life = 8.0, 72.0

gd = _calibrate_predictions(baseline_death, baseline_life, fg)
bd = _calibrate_predictions(baseline_death, baseline_life, fb)

print('Healthy baseline-> adjusted:', gd)
print('Unhealthy baseline-> adjusted:', bd)
