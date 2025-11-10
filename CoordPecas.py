import cv2
import numpy as np
import os # Usado para limpar o console

# Lista global para armazenar os pontos de clique
pontos_pixel_clicados = []
imagem_calibracao = None

# --- 1. SEÇÃO DE CALIBRACAO INTERATIVA (Funções) ---

def clique_callback(event, x, y, flags, param):
    """Função de callback do mouse."""
    # Esta declaração 'global' está CORRETA, pois está DENTRO da função
    global pontos_pixel_clicados, imagem_calibracao 
    
    if event == cv2.EVENT_LBUTTONDOWN and len(pontos_pixel_clicados) < 3:
        pontos_pixel_clicados.append((x, y))
        cv2.circle(imagem_calibracao, (x, y), 5, (0, 0, 255), -1)
        
        if len(pontos_pixel_clicados) == 1:
            print(f"  Ponto 1/3 (Origem 0,0) definido em pixel: {x, y}")
        elif len(pontos_pixel_clicados) == 2:
            print(f"  Ponto 2/3 (Eixo X - 600,0) definido em pixel: {x, y}")
        elif len(pontos_pixel_clicados) == 3:
            print(f"  Ponto 3/3 (Eixo Y - 0,400) definido em pixel: {x, y}")
            print("\n✅ Calibração concluída! Pressione qualquer tecla...")

def redimensionar_imagem(img, largura_desejada):
    """Redimensiona a imagem para caber na tela mantendo a proporção."""
    try:
        proporcao = largura_desejada / img.shape[1]
        altura_desejada = int(img.shape[0] * proporcao)
        dimensoes = (largura_desejada, altura_desejada)
        return cv2.resize(img, dimensoes, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Erro ao redimensionar: {e}")
        return img # Retorna a imagem original se falhar

# --- 2. INICIAR WEBCAM E CAPTURAR FRAME PARA CALIBRAR ---

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("❌ Erro: Não foi possível abrir a webcam.")
    print("Tente alterar o '0' em cv2.VideoCapture(0) para 1 ou 2.")
    exit()

print("--- 🤖 AGUARDANDO CALIBRACAO ---")
print("Webcam aberta. Posicione a câmera.")
print("Pressione 'c' para capturar um frame e iniciar a calibração.")
print("Pressione 'ESC' para sair.")

frame_calibracao = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da webcam.")
        break
        
    frame_exibicao = redimensionar_imagem(frame, 1000)
    cv2.imshow("CALIBRACAO - Pressione 'c' para capturar", frame_exibicao)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        frame_calibracao = frame.copy() # Congela o frame
        print("\nFrame capturado! Iniciando calibração...")
        cv2.destroyWindow("CALIBRACAO - Pressione 'c' para capturar")
        break
    elif key == 27: # ESC
        print("Saindo...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- 3. EXECUTAR CALIBRACAO INTERATIVA (no frame congelado) ---

# *** INÍCIO DA CORREÇÃO ***
# A linha 'global imagem_calibracao' foi REMOVIDA daqui, pois é desnecessária e incorreta no escopo global.
# A variável 'imagem_calibracao' já foi definida como 'None' no topo do script.
# *** FIM DA CORREÇÃO ***
imagem_calibracao = redimensionar_imagem(frame_calibracao, 1000)

cv2.namedWindow("CALIBRACAO - Clique em 3 pontos")
cv2.setMouseCallback("CALIBRACAO - Clique em 3 pontos", clique_callback)

print("  1. Clique na ORIGEM (0, 0)")
print("  2. Clique no ponto do EIXO X (600, 0)")
print("  3. Clique no ponto do EIXO Y (0, 400)")
print("\nPressione qualquer tecla na janela após os 3 cliques.")

while len(pontos_pixel_clicados) < 3:
    cv2.imshow("CALIBRACAO - Clique em 3 pontos", imagem_calibracao)
    if cv2.waitKey(1) & 0xFF == 27: # Sair com 'ESC'
        print("Calibração cancelada.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
cv2.imshow("CALIBRACAO - Clique em 3 pontos", imagem_calibracao)
cv2.waitKey(0)
cv2.destroyWindow("CALIBRACAO - Clique em 3 pontos")

# --- 4. CÁLCULO DA TRANSFORMAÇÃO ---
escala = frame_calibracao.shape[1] / imagem_calibracao.shape[1]
pts_pixel = np.float32(pontos_pixel_clicados) * escala

pts_desenho = np.float32([
    [0, 0],      # Ponto 1 (Origem)
    [600, 0],    # Ponto 2 (Eixo X)
    [0, 400]     # Ponto 3 (Eixo Y)
])

try:
    matriz_transformacao = cv2.getAffineTransform(pts_pixel, pts_desenho)
    print("✅ Matriz de calibração calculada com sucesso.")
    print("\n--- 🚀 INICIANDO DETECCAO EM TEMPO REAL ---")
    print("Pressione 'q' na janela de vídeo para sair.")
except cv2.error as e:
    print(f"❌ Erro ao calcular a matriz: {e}")
    cap.release()
    exit()

# --- 5. LOOP PRINCIPAL DE DETECÇÃO (EM TEMPO REAL) ---

MIN_AREA = 1500 
MIN_SOLIDITY = 0.9
LIMIAR_INTENSIDADE_DESTINO = 180 # Ajuste se necessário

while True:
    ret, frame = cap.read()
    if not ret:
        print("Feed da webcam perdido.")
        break

    pecas_encontradas = []
    peca_contador = 1
    imagem_resultado = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 

    kernel = np.ones((3,3),np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations = 1)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations = 1)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(contorno)
        (center_px_raw, (w_px, h_px), angle_px_raw) = rect

        if h_px == 0 or w_px == 0: continue
        
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue 
        solidity = float(area) / hull_area
        
        if solidity < MIN_SOLIDITY:
            continue 

        cX_pixel = int(center_px_raw[0])
        cY_pixel = int(center_px_raw[1])
        ponto_pixel_np = np.float32([[[cX_pixel, cY_pixel]]])
        ponto_desenho = cv2.transform(ponto_pixel_np, matriz_transformacao)
        cX_desenho = ponto_desenho[0][0][0]
        cY_desenho = ponto_desenho[0][0][1]

        box_pixels = cv2.boxPoints(rect)
        pt1_px = box_pixels[0]; pt2_px = box_pixels[1]
        pontos_px_np = np.float32([[pt1_px, pt2_px]])
        pontos_mundo = cv2.transform(pontos_px_np, matriz_transformacao)
        pt1_mundo = pontos_mundo[0][0]; pt2_mundo = pontos_mundo[0][1]
        
        delta_Y_mundo = pt2_mundo[1] - pt1_mundo[1]
        delta_X_mundo = pt2_mundo[0] - pt1_mundo[0]
        angle_rad = np.arctan2(delta_Y_mundo, delta_X_mundo)
        angle_deg = np.degrees(angle_rad)
        
        if w_px < h_px: angle_deg += 90
        angle_deg = angle_deg % 180

        mascara = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        intensidade_media = cv2.mean(gray, mask=mascara)[0]

        tipo_objeto = ""; id_objeto = ""
        
        if intensidade_media > LIMIAR_INTENSIDADE_DESTINO: 
            tipo_objeto = "Destino"; id_objeto = "Destino" 
            cor_contorno = (0, 255, 255) # Amarelo
        else: 
            tipo_objeto = "Peca"; id_objeto = f"Peca{peca_contador}" 
            peca_contador += 1; cor_contorno = (0, 255, 0) # Verde

        peca = {
            "id": id_objeto, "centro_pixel": (cX_pixel, cY_pixel),
            "coords_mundo": (round(cX_desenho, 2), round(cY_desenho, 2)),
            "angulo_mundo": round(angle_deg, 2), "tipo": tipo_objeto
        }
        pecas_encontradas.append(peca)

        box_pixels_int = np.int0(box_pixels)
        cv2.drawContours(imagem_resultado,[box_pixels_int],0,cor_contorno,2)
        cv2.circle(imagem_resultado, (cX_pixel, cY_pixel), 5, (0, 0, 255), -1)
        
        texto_x = f"{peca['coords_mundo'][0]:.1f}"
        texto_y = f"{peca['coords_mundo'][1]:.1f}"
        texto_a = f"{peca['angulo_mundo']:.1f}"
        texto = f"{id_objeto}: ({texto_x}, {texto_y}) A: {texto_a}"
        
        cv2.putText(imagem_resultado, texto, (cX_pixel - 70, cY_pixel - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- 6. Exibir Resultados (Console e Janelas) ---

    os.system('cls' if os.name == 'nt' else 'clear') 
    print("--- 🤖 RESULTADOS PARA ROBOTSTUDIO (AO VIVO) ---")
    
    if not pecas_encontradas:
        print("Nenhuma peça ou alvo retangular encontrado.")
    else:
        pecas_ordenadas = sorted(pecas_encontradas, key=lambda p: p['tipo'] == 'Destino')
        for peca in pecas_ordenadas:
            print(f"ID: {peca['id']}")
            print(f"  Tipo: {peca['tipo']}")
            print(f"  Coords (Mundo): X = {peca['coords_mundo'][0]}, Y = {peca['coords_mundo'][1]}")
            print(f"  Angulo (Mundo): {peca['angulo_mundo']} graus")
            print(f"  (Coords Pixel): u = {peca['centro_pixel'][0]}, v = {peca['centro_pixel'][1]}\n")
    print("-------------------------------------------------")
    print("Pressione 'q' na janela de vídeo para SAIR.")

    cv2.imshow("Binaria", redimensionar_imagem(binaria, 800))
    cv2.imshow("Resultado Final", redimensionar_imagem(imagem_resultado, 800))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Encerrando...")
        break

# --- 7. Limpeza Final ---
cap.release()
cv2.destroyAllWindows()