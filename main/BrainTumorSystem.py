import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Biblioteca YOLO do ultralytics

def carregar_modelo(caminho_modelo):
    """
    Carrega o modelo YOLO a partir do arquivo especificado.
    """
    modelo = YOLO(caminho_modelo)  # Carrega o modelo YOLO
    return modelo

def carregar_imagem(caminho_imagem):
    """
    Carrega e prepara a imagem para análise.
    """
    img = cv2.imread(caminho_imagem)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
    return img_rgb, img

def realizar_deteccao(modelo, img):
    """
    Realiza a detecção de objetos na imagem utilizando o modelo YOLO carregado.
    """
    resultados = modelo(img)  # Faz a inferência no modelo
    return resultados

def exibir_resultados(img_rgb, resultados):
    """
    Exibe a imagem com as caixas de detecção.
    """
    plt.imshow(img_rgb)
    ax = plt.gca()

    for resultado in resultados:
        for box in resultado.boxes.xyxy:  # Para cada caixa detectada
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

    plt.show()

def inserir_imagem():

    # Cria a janela principal
    janela = tk.Tk()
    janela.withdraw() # Oculta a janela principal

    # Abre a janela de diálogo para a seleção da imagem
    caminho = filedialog.askopenfilename()

    # Fecha a janela principal
    janela.destroy() 

    # Retorna o caminho da imagem escolhida pelo usuário
    return caminho


def main():
    caminho_modelo = '../best_model/my_model.pt'  # Substitua pelo caminho do seu modelo YOLO
    caminho_imagem = inserir_imagem()  # Interface para inserir imagem

    modelo = carregar_modelo(caminho_modelo)

    try:
        img_rgb, img = carregar_imagem(caminho_imagem)
        resultados = realizar_deteccao(modelo, img)
        exibir_resultados(img_rgb, resultados)
    except:
        print("Erro ao carregar imagem!")


if __name__ == '__main__':
    main()
