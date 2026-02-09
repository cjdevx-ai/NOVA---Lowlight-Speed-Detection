import roboflow

rf = roboflow.Roboflow(api_key="UMn9vdgXLxL0XGiK66mz")
model = rf.workspace().project("nova-v2-5000").version("3").model
model.download() # Downloads 'weights.pt' to your local folder