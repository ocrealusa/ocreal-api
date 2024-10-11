install_deps:
	@pip install -r requirements.txt

run_app:
	@uvicorn main:app --reload