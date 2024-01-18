from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField


class UrlForm(FlaskForm):
    url = StringField("Enter URL")
