# SASN - Simple Abstract Syntax Parser

## Használati útmutató

* `sudo systemctl start docker`
* `sudo docker build -t sasn .`
* `sudo docker build --no-cache -t sasn .`
* `sudo docker run -it --rm sasn`
* `python main.py --help`
* `python main.py grammargen --help`
* `python main.py grammargen mlforse`
* `python main.py preprocess mlforse train.json train.pickle`
* `python main.py train mlforse train.pickle model.pickle`
* `python main.py query mlforse model.pickle "What is the sum of the grades of the students in class?"`
