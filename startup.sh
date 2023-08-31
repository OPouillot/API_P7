
apt-get install libgomp1

gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app