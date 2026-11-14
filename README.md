# camfundo

Remoção de fundo da webcam com IA, em tempo real, para Windows. Uma alternativa ao NVIDIA Broadcast que roda em GPUs **AMD, Intel e NVIDIA** (via DirectML) ou na CPU.

O vídeo processado sai por uma câmera virtual, então funciona em qualquer app: Discord, Meet, Zoom, Teams, OBS etc.

## Recursos

- Remoção de fundo com *video matting* (bordas suaves, com transparência gradual no cabelo) e consistência entre frames
- Fundo substituído por **cor sólida** (verde para chroma key por padrão), **imagem** ou **desfoque**
- Roda na GPU via DirectML (AMD, Intel, NVIDIA) ou na CPU
- Controles para equilibrar qualidade, FPS e consumo de GPU/CPU
- Label com coração batendo na parte de baixo do vídeo, que muda de cor e de velocidade conforme o BPM, informado manualmente ou **estimado pela câmera** (rPPG)
- Modo sem janela e diagnóstico de desempenho (`--profile`)

## Como funciona

```
Webcam → Robust Video Matting (ONNX Runtime + DirectML) → composição do fundo → OBS Virtual Camera → seu app
```

O modelo é o [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), baixado automaticamente na primeira execução. A saída vai para a **OBS Virtual Camera** por meio da biblioteca `pyvirtualcam`.

> O programa **substitui** o fundo; ele não gera vídeo com transparência. Se você precisa de transparência real (por exemplo, colocar sua imagem sobre um jogo no OBS), use o fundo verde padrão e aplique o filtro **Chroma Key** do OBS.

## Requisitos

- Windows 10 ou 11
- Webcam
- [OBS Studio](https://obsproject.com/) (fornece a câmera virtual)
- [Python](https://www.python.org/downloads/) **3.10 a 3.12** (recomendado: 3.12). Versões muito novas do Python podem não ter pacotes do `onnxruntime-directml`.
- OpenCV 4.x (o `requirements.txt` já fixa `opencv-python>=4.8,<5`; necessário para o modelo de rosto do `--bpm-camera`)
- GPU compatível com DirectX 12 (opcional: dá para usar a CPU com `--device cpu`)

## Instalação

**1. OBS Studio.** Instale e abra uma vez (isso registra a câmera virtual). Pode fechar depois.

**2. Python.** Instale a versão 3.12 marcando **"Add python.exe to PATH"**. Feche e abra o terminal depois.

**3. Projeto.** Baixe este repositório (Code → Download ZIP) ou clone:

```powershell
git clone https://github.com/SEU-USUARIO/camfundo.git
cd camfundo
```

**4. Ambiente virtual e dependências.** No PowerShell, dentro da pasta do projeto:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**5. Descubra o número da sua webcam.**

```powershell
venv\Scripts\python.exe camfundo.py --list-cameras
```

O comando abre cada câmera por cerca de 1,5 s, mostra uma prévia com o número e informa no console se a imagem está normal ou preta (câmeras virtuais costumam aparecer pretas). Anote o número da câmera correta.

**6. Primeira execução.**

```powershell
venv\Scripts\python.exe camfundo.py --camera 1 --preview
```

Na primeira vez, o modelo é baixado para a pasta `models/`. No console, a linha `Provedor em uso: DmlExecutionProvider` indica que a IA está na GPU. Se aparecer `CPUExecutionProvider`, está na CPU.

> Nos exemplos abaixo, `python` significa o Python do ambiente virtual. Ative com `venv\Scripts\activate` ou escreva `venv\Scripts\python.exe` no lugar. Se o PowerShell bloquear o `activate`, rode antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Uso

Com preview (janela para conferir o resultado):

```powershell
python camfundo.py --camera 1 --preview
```

Sem janela, só enviando para a câmera virtual (encerre com **Ctrl+C**):

```powershell
python camfundo.py --camera 1
```

Exemplos de fundo:

```powershell
# Imagem de fundo
python camfundo.py --camera 1 --mode image --image "D:/Imagens/fundo.png"

# Desfoque
python camfundo.py --camera 1 --mode blur --blur 40

# Cor sólida personalizada (hex)
python camfundo.py --camera 1 --mode color --color 000000
```

### Usando nos aplicativos

1. Deixe o `camfundo` rodando.
2. No app (Discord, Meet, Zoom, Teams...), escolha a câmera **OBS Virtual Camera**.
3. Para chroma key no OBS: adicione a **OBS Virtual Camera** como fonte, aplique o filtro **Chroma Key** e escolha o verde.

## Opções

Todas as opções são passadas na linha de comando. As opções de liga/desliga ficam **desativadas** por padrão e só funcionam se você escrevê-las.

### Câmera e saída

| Opção | Padrão | Descrição |
|---|---|---|
| `--camera N` | `0` | Índice da webcam. Use `--list-cameras` para descobrir. |
| `--width N` | `1280` | Largura pedida à webcam. O driver decide o que entrega, e o programa usa a resolução real. |
| `--height N` | `720` | Altura pedida à webcam. |
| `--fps N` | `30` | Limite de quadros processados por segundo. Também é pedido à webcam e à câmera virtual. Só reduz: não aumenta o que a câmera entrega. |
| `--mirror` | desligado | Espelha a imagem. |
| `--preview` | desligado | Mostra uma janela com o resultado. |
| `--no-vcam` | desligado | Não envia para a câmera virtual (útil para testes). Sem `--preview`, o preview é ligado automaticamente. |
| `--list-cameras` | — | Testa os índices de 0 a 5, mostra uma prévia de cada e encerra. |

### Fundo

| Opção | Padrão | Descrição |
|---|---|---|
| `--mode` | `color` | `color` (cor sólida), `image` (imagem), `blur` (desfoque) ou `none` (sem efeito). |
| `--color HEX` | `00B140` | Cor do fundo no modo `color` (verde de chroma key por padrão). |
| `--image CAMINHO` | — | Imagem de fundo para `--mode image`. É ajustada para cobrir toda a tela, cortando as bordas sem esticar. Use aspas se o caminho tiver espaços. |
| `--blur N` | `30` | Intensidade do desfoque no modo `blur`. |

### IA e qualidade

| Opção | Padrão | Descrição |
|---|---|---|
| `--model` | `mobilenet` | `mobilenet` (leve e rápido) ou `resnet50` (mais preciso e mais pesado). |
| `--device` | `dml` | `dml` usa a GPU via DirectML; `cpu` força a CPU. |
| `--infer-width N` | `640` | Largura da imagem analisada pela IA (arredondada para múltiplo de 16). Maior = bordas melhores (cabelo, dedos), porém mais custo. |
| `--ratio X` | automático | `downsample_ratio` do modelo. Maior = mais detalhe e mais custo. Normalmente não precisa mexer. |
| `--edge X` | `0.08` | Limpeza das bordas da máscara (0 a 0.3). Valores maiores removem a "névoa" ao redor de você. |
| `--shrink N` | `0` | Encolhe a máscara em N pixels. Ajuda a eliminar halo em volta da pessoa. |

### Desempenho e consumo

| Opção | Padrão | Descrição |
|---|---|---|
| `--paralelo` | desligado | Roda a IA em uma thread separada. A saída fica fluida e cada frame é composto com a máscara mais recente, que pode atrasar cerca de um frame. |
| `--ia-fps N` | `0` (sem limite) | Limita quantas vezes por segundo a IA roda. **Só vale com `--paralelo`.** Reduz o uso de GPU, mas deixa a máscara mais velha. |
| `--fluxo` | desligado | Compensa o movimento da máscara velha usando fluxo óptico (CPU). Útil com `--ia-fps` baixo ou `--skip`. |
| `--skip N` | `1` | Roda a IA a cada N frames e reaproveita a máscara nos outros. Não é usado com `--paralelo`. Pode causar borda atrasada e sensação de engasgo. |
| `--threads N` | `0` (automático) | Limita as threads de CPU usadas pelo OpenCV e pelo ONNX Runtime. |
| `--prioridade` | `normal` | `baixa` faz o programa ceder CPU para jogos e outros apps. |

### Label de coração (BPM)

Desenha, embaixo e centralizado no vídeo de saída, um coração que bate no ritmo do BPM, com o número ao lado. Quanto mais baixo o BPM, mais gelado (azul) e lento; quanto mais alto, mais quente (vermelho) e rápido.

O BPM pode ser **manual** (`--bpm-teste`) ou **medido pela câmera** (`--bpm-camera`).

| Opção | Padrão | Descrição |
|---|---|---|
| `--coracao` | desligado | Liga o label. Usa 75 BPM manual se nenhuma outra opção de BPM for informada. |
| `--bpm-teste N` | `75` | BPM manual mostrado pelo coração. Informar este valor já liga o label. |
| `--bpm-camera` | desligado | Mede o BPM pela câmera (rPPG) e mostra no coração. Liga o label. |
| `--bpm-snr X` | `0.8` | Confiança mínima para aceitar uma leitura. Menor = mais sensível, porém com mais risco de erro. |
| `--bpm-debug` | desligado | No preview, mostra o rosto detectado, as regiões medidas e a confiança do sinal. |
| `--bpm-min N` | `50` | BPM que corresponde à cor mais gelada. |
| `--bpm-max N` | `120` | BPM que corresponde à cor mais quente. |
| `--bpm-sem-numero` | desligado | Mostra só o coração, sem o número. |

Exemplos:

```powershell
# BPM manual
python camfundo.py --camera 1 --preview --bpm-teste 80

# BPM medido pela câmera, com as regiões medidas visíveis no preview
python camfundo.py --camera 1 --preview --bpm-camera --bpm-debug
```

#### Como o BPM é medido

A técnica se chama fotopletismografia remota (rPPG): a cada batimento, a quantidade de sangue na pele muda um pouco, e isso altera de leve a cor do rosto.

1. Um detector de rosto (YuNet, do OpenCV Zoo) encontra o rosto. O modelo é baixado automaticamente na primeira vez com `--bpm-camera`.
2. É medida a cor média de três regiões (testa, bochecha esquerda e bochecha direita) em cada frame, usando a imagem original da câmera (antes de trocar o fundo).
3. Os últimos ~15 segundos de cada região passam pelo método POS, que combina R, G e B para destacar o pulso e reduzir o efeito de movimento e de variação de luz. Picos isolados (movimento, piscadas de luz) são limitados, e as regiões cujo sinal tem um pico mais nítido pesam mais no resultado.
4. Uma FFT encontra a frequência dominante entre 45 e 180 bpm, e um índice de confiança valida o resultado. Picos colados nas bordas dessa faixa são descartados, porque costumam ser ruído de luz, não pulso.
5. A primeira leitura só é aceita depois de 3 estimativas consecutivas concordando. A partir daí o programa **trava** nessa frequência e só acompanha mudanças graduais (até ±15 bpm por vez), em vez de saltar para outro pico de ruído (por exemplo, de 80 para 50). Se o sinal piorar, o último valor é mantido por até 10 segundos; depois disso o número some e aparece `sinal fraco`. Ao voltar, durante 60 segundos só são aceitos valores plausíveis (até ±25 bpm do último). Na dúvida, o programa prefere não mostrar número a mostrar um valor absurdo.

O coração mostra o estado da medição:

| Mensagem | Significado |
|---|---|
| `medindo...` | Juntando dados. A primeira leitura leva cerca de 12 a 15 segundos com o rosto visível. |
| `sem rosto` | Nenhum rosto foi detectado há mais de 1 segundo. |
| `sinal fraco` | O sinal não tem confiança suficiente (movimento, pouca luz, luz piscando). |
| `erro no rosto` / `sem modelo` | O detector de rosto não está funcionando (veja a solução de problemas). |

Dicas para uma leitura melhor:

- **Luz estável e de frente.** Evite lâmpadas que piscam (LED barato) e telas ou luzes que mudam de cor perto do rosto, pois isso cria "pulsos" falsos.
- **Fique parado** e com o rosto inteiro no quadro, a uma distância em que ele ocupe uma boa parte da imagem.
- **Desative a exposição e o balanço de branco automáticos da webcam**, se o app ou o driver dela permitir, porque eles alteram a cor da imagem por conta própria e atrapalham a medição.
- Rode com `--bpm-debug`: os retângulos amarelos devem ficar sobre a testa e as bochechas (sem cabelo, óculos ou barba cobrindo), e a linha de texto mostra o valor bruto, a confiança e se o programa está `travado` ou `buscando`.

> **Aviso:** é uma estimativa por vídeo, feita para curiosidade e demonstração. **Não é um dispositivo médico** e não deve ser usada para decisões de saúde. Em movimento, com pouca luz ou com a câmera ajustando a imagem sozinha, o valor pode errar ou não aparecer. Compare com um relógio ou oxímetro antes de confiar nos números.

### Diagnóstico

| Opção | Padrão | Descrição |
|---|---|---|
| `--profile` | desligado | Imprime a cada segundo o FPS e o tempo médio (ms) por etapa. |

Exemplo de saída do `--profile`:

```
24.0 FPS | leitura 17.5 ms | ia 13.0 ms | fluxo 0.0 ms | composicao 6.0 ms | overlay 0.0 ms | bpm 0.0 ms | envio 5.0 ms
```

- **leitura:** tempo esperando a webcam entregar o frame. Se for alto mesmo sem IA (`--mode none`), o limite é a câmera.
- **ia:** tempo da inferência. Com `--paralelo`, fica perto de zero e aparece `IA N/s`, que indica quantas vezes por segundo a IA rodou.
- **fluxo:** tempo da compensação de movimento (`--fluxo`).
- **composicao**, **overlay**, **bpm** e **envio:** mistura do fundo, desenho do label de coração, medição do BPM (`--bpm-camera`) e envio para a câmera virtual.

## Atalhos no preview

Com a janela de preview aberta (`--preview`):

| Tecla | Ação |
|---|---|
| `b` | Fundo desfocado |
| `c` | Cor sólida |
| `i` | Imagem de fundo (só funciona se o programa foi iniciado com `--image`) |
| `n` | Sem efeito |
| `+` / `-` | Aumenta / diminui o BPM de teste em 5 (só com `--bpm-teste`) |
| `h` | Mostra ou esconde o coração |
| `q` ou `Esc` | Sair |

## Receitas

**Equilíbrio (bom ponto de partida):**
```powershell
python camfundo.py --camera 1 --width 1280 --height 720 --fps 24 --infer-width 540
```

**Qualidade máxima nas bordas:**
```powershell
python camfundo.py --camera 1 --model resnet50 --infer-width 960 --shrink 2 --edge 0.12
```

**Economizar GPU/CPU (jogando ou transmitindo ao mesmo tempo):**
```powershell
python camfundo.py --camera 1 --fps 20 --infer-width 480 --threads 2 --prioridade baixa
```

**Reduzir a frequência da IA mantendo a máscara alinhada:**
```powershell
python camfundo.py --camera 1 --paralelo --ia-fps 12 --fluxo --infer-width 960
```

Dica: para gastar menos recurso sem prejudicar a qualidade da máscara, prefira reduzir `--fps` ou `--infer-width`. Rodar a IA menos vezes (`--skip`, `--ia-fps`) deixa a máscara velha em relação ao vídeo, o que aparece como borda "arrastando" quando você se mexe.

## Arquivos `.bat`

| Arquivo | O que faz |
|---|---|
| `iniciar_camfundo.bat` | Abre o programa com um clique, usando o Python do `venv`. |
| `iniciar_camfundo_oculto.bat` | Abre sem janela de console (via `pythonw`) e com prioridade baixa. |
| `parar_camfundo.bat` | Encerra a instância que está rodando em modo oculto. |

Edite os `.bat` com o Bloco de Notas para mudar a câmera ou as opções (a linha que começa com `python` ou `pythonw`). Mantenha-os na mesma pasta do `camfundo.py` e do `venv`.

### Modo sem janela

No `pythonw` não existe console, então:

- mensagens e erros vão para `camfundo.log`, na pasta do programa;
- o número do processo é gravado em `camfundo.pid` enquanto o programa roda;
- para encerrar, use o `parar_camfundo.bat`, ou finalize o processo `pythonw.exe` no Gerenciador de Tarefas.

O modo sem janela não reduz o consumo de CPU ou GPU: ele só esconde o console.

## Solução de problemas

**`pip` não é reconhecido.** O Python não está no PATH. Reinstale marcando "Add python.exe to PATH" e reabra o terminal, ou use `py -m pip`.

**`No matching distribution found for onnxruntime-directml`.** Seu Python é novo demais. Instale o Python 3.12 e recrie o ambiente virtual (`py -3.12 -m venv venv`).

**`ModuleNotFoundError: No module named 'cv2'`.** O programa rodou fora do ambiente virtual, ou a pasta `venv` foi movida. Use `venv\Scripts\python.exe camfundo.py ...` ou recrie o ambiente:

```powershell
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Tela preta e 1 FPS.** Provavelmente o índice da câmera é uma câmera virtual (OBS Virtual Camera, DroidCam, Iriun...). Rode `--list-cameras` e use o número da webcam real.

**`Falha ao abrir a câmera virtual`.** Instale o OBS Studio e abra-o uma vez. Também confira se outro programa não está usando a câmera virtual.

**O console mostra `CPUExecutionProvider`.** O DirectML não está sendo usado. Confirme que só o `onnxruntime-directml` está instalado (o pacote `onnxruntime` comum instalado junto pode atrapalhar):

```powershell
venv\Scripts\python.exe -m pip uninstall -y onnxruntime
venv\Scripts\python.exe -m pip install --force-reinstall onnxruntime-directml
```

**`Falha ao iniciar o detector de rosto` ou `erro no rosto` (com `--bpm-camera`).** O modelo de rosto foi feito para o OpenCV 4.x, e o OpenCV 5 pode não ser compatível. Instale a versão 4:

```powershell
venv\Scripts\python.exe -m pip install "opencv-python>=4.8,<5"
```

Se o problema for o download do modelo, baixe `face_detection_yunet_2023mar.onnx` do repositório [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) e coloque na pasta `models/`.

**O BPM fica em `sinal fraco` ou `medindo...` para sempre.** Aumente e estabilize a luz, fique parado de frente para a câmera e rode com `--bpm-debug` para ver se as regiões (retângulos amarelos) estão sobre a testa e as bochechas. Se o sinal nunca passa da confiança mínima, teste `--bpm-snr 0.6`, sabendo que isso aumenta o risco de leituras erradas.

**FPS preso em 24 (ou 15).** Muitas webcams reduzem o FPS em ambientes escuros por causa da exposição automática. Aumente a luz, desative opções como "baixa luminosidade" ou "taxa de quadros dinâmica" no app da webcam e teste `--mode none --profile`: se o FPS continuar baixo sem IA, o limite é a câmera.

**Halo ou contorno em volta de mim.** Use `--shrink 2` e/ou `--edge 0.12`.

**A borda "arrasta" quando me mexo.** A máscara está velha. Use `--ia-fps` maior, adicione `--fluxo`, ou abandone `--skip` e `--paralelo`.

**Alto uso de GPU.** Reduza `--infer-width`, reduza `--fps`, use o modelo `mobilenet` (padrão) e evite `--skip`/`--ia-fps` baixos.

## Créditos

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), de Shanchuan Lin et al. (*Robust High-Resolution Video Matting with Temporal Guidance*, WACV 2022), licenciado sob GPL-3.0. O modelo não é distribuído neste repositório: é baixado do repositório original na primeira execução.
- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo, licença MIT), para detectar o rosto na medição de BPM. O modelo também é baixado na primeira execução.
- Método POS para rPPG: W. Wang et al., *Algorithmic Principles of Remote PPG*, IEEE Trans. Biomedical Engineering, 2017.
- [ONNX Runtime](https://onnxruntime.ai/) (DirectML), [OpenCV](https://opencv.org/), [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) e [OBS Studio](https://obsproject.com/).
