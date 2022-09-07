import pandas as pd #biblioteca para trabalhar com csv
import numpy as np #auxiliar matematico das outras libs
import matplotlib.pyplot as plt #gerador de graficos

from sklearn.preprocessing import StandardScaler #
from sklearn.model_selection import train_test_split #divide a base de dados para treino e teste
from sklearn.metrics import r2_score #métrica de validaçaõ
from sklearn.linear_model import  SGDRegressor #descida do gradiente estocastica
from sklearn.neural_network import MLPRegressor 
 
# dataset = pd.read_csv(".auto-mpg.csv")

dataset = pd.read_csv("..\data\Cov19_Datasets\Cov19_Control_x_Covid\FeatureVectors_Cov19_Control_x_Covid_Fitting.csv")
classes = dataset[3:4]
dataset = dataset[5:]

x = dataset[dataset.columns.values[1:]].T
y = classes[dataset.columns.values[1:]].T
y = np.concatenate(y.values, axis=0).astype(int)

# x =  dataset[["weight"]] #entrada
# y =  dataset[["mpg"]] #rótulos (saída esperada)

# x *= 0.453592
# y *= 0.415144

# print(x)
# print(y)

#Normaliza os valores para entre [0,1]
escala = StandardScaler()
escala.fit(x)
xNorm = escala.transform(x)
# print(xNorm)

xNormTrain, xNormTest, yTrain, yTest  = train_test_split(xNorm, y, test_size=0.3)
# print(xNormTest, xNormTrain, yTrain, yTest)

#MLP
rna = MLPRegressor(max_iter=2000, tol=0.00000001, 
                   learning_rate_init = 0.001, solver="adam", activation="relu", 
                   learning_rate="constant", verbose=2)
rna.fit(xNormTrain, yTrain)

# #regressão linear
# reglinear = SGDRegressor(max_iter=2000, tol=0.0000001, eta0=0.1, learning_rate="constant", verbose=2)
# reglinear.fit(xNormTrain, yTrain)

y_rna_predict = rna.predict(xNormTest)
r2_rna = r2_score(yTest, y_rna_predict)
print(yTest, y_rna_predict)
print("R2 RNA:", r2_rna)

# y_rl_predict = reglinear.predict(xNormTest)
# r2_rl = r2_score(yTest, y_rl_predict)
# print("R2 RL:", r2_rl)

'''
    "editor.fontFamily": "IBM Plex Mono, Monaco, Courier New, monospace",
    "editor.fontSize": 15,
    "editor.lineHeight": 26,
    "editor.tabSize": 2,
    "editor.tabCompletion": "on",
    "editor.wordWrap": "on",
    "workbench.activityBar.visible": true,
    "liveServer.settings.donotShowInfoMsg": true,
    "liveServer.settings.donotVerifyTags": true,
    "explorer.confirmDragAndDrop": false,
    "diffEditor.renderSideBySide": true,
    "editor.formatOnSave": true,
    "html.autoClosingTags": false,
    "editor.colorDecorators": false,
    "editor.autoClosingBrackets": "always",
    "editor.autoClosingQuotes": "always",
    "prettier.singleQuote": true,
    "files.associations": {
      "*.js": "javascriptreact"
    },
    "editor.minimap.renderCharacters": false,
    "breadcrumbs.enabled": false,
    "telemetry.enableCrashReporter": false,
    "telemetry.enableTelemetry": false,
    "color-highlight.markerType": "dot-before",
    "editor.renderWhitespace": "selection",
    "workbench.statusBar.visible": true,
    "prettier.trailingComma": "all",
    "[javascriptreact]": {
      "editor.defaultFormatter": "esbenp.prettier-vscode"
    },
    "[json]": {
      "editor.defaultFormatter": "esbenp.prettier-vscode"
    },
    "explorer.sortOrder": "default",
    "window.title": "${activeEditorMedium}${separator}${rootName}",
    "window.newWindowDimensions": "inherit",
    "workbench.colorTheme": "Origamid Next",
    "workbench.iconTheme": "origamid-next-icons",
    "html.format.wrapAttributes": "auto",
    "html.format.wrapLineLength": 0,
    "[html]": {
      "editor.defaultFormatter": "vscode.html-language-features"
    },
'''