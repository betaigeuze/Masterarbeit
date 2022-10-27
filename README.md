# RaFoView
## Random Forest View
An educational dashboard built as a proof of concept for investigating Random Forest models.

## Manuscript
This project was built for the master's thesis found under:
PLACEHOLDER

## Dependencies
This project relies on the following main packages:
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Vega-Altair](https://altair-viz.github.io/)
- [NetworkX](https://networkx.org/)
- [PyGraphviz](https://pygraphviz.github.io/)

## Data
The dashboard builds 2 example use cases using the Iris and the Digits dataset from the Scikit-learn package.

## Code
|Folder|Description|
|:---|:---|
|[src/dashboardv1/](src/dashboardv1/)|Contains all of the necessary python code
|[src/dashboardv1/images/](src/dashboardv1/images/)|Contains the images used in the dashboard
|[src/dashboardv1/pickle/](src/dashboardv1/pickle/)|Contains the precomputed pickle files for the pairwise distance matrix
|[src/dashboardv1/text/](src/dashboardv1/text/)|Contains the markdown files used in the dashboard
|[tests/](/tests/)|Is currently empty
|[.streamlit/](/.streamlit)|Contains the *.toml file for the custom Streamlit theme
|[.vscode/](/.vscode)|Contains the launch.json and settings.json for VSCode development


## Installation
To run the code, clone the repository and install the dependencies by installing any environment manager that you prefer. You can then use the requirements.txt file to install all necessary dependencies. The code was developed using pipenv, but you can use any environment manager you prefer.
For more information on how to use pipenv:
https://pipenv.pypa.io/en/latest/basics/#example-pipenv-workflow  

## Usage
After installing all necessary dependencies, you can run the dashboard by running the following command in the terminal:
```
streamlit run src/dashboardv1/st_dashboard.py
```
Streamlit should display a link to the dashboard in the terminal. If it does not, you can also access the dashboard by navigating to http://localhost:8501 in your browser.

## License

Licensed under the MIT Licence,([LICENSE](./LICENSE))
```
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
```