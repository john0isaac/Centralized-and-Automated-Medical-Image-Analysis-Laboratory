# Centralized-and-Automated-Medical-Image-Analysis-Laboratory

## Development Setup
1. **Download the project starter code locally**
```
git clone https://github.com/John0Isaac/Centralized-and-Automated-Medical-Image-Analysis-Laboratory.git
cd Centralized-and-Automated-Medical-Image-Analysis-Laboratory
```

2. **Before and After editing your code, Use the commands below:**
before editing anything pull new changes from GitHub.
```
git pull
```
Once you are done editing, you can push the local repository to your Github account using the following commands.
```
git add .
git commit -m "your comment message"
git push
```

3. **Initialize and activate a virtualenv using:**
```
python -m virtualenv venv
source venv/bin/activate
```
>**Note** - In Windows, the `venv` does not have a `bin` directory. Therefore, you'd use the analogous command shown below:
```
source venv/Scripts/activate
deactivate
```

4. **Install the dependencies:**
```
pip install -r requirements.txt
```

5. **Run the development server:**
```
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=true
flask run --reload
```

6. **Verify on the Browser**<br>
Navigate to project homepage [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000](http://localhost:5000)