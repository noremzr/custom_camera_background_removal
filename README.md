# camfundo

Remoção de fundo da webcam com IA, em tempo real, para Windows. Uma alternativa ao NVIDIA Broadcast que roda em GPUs **AMD, Intel e NVIDIA** (via DirectML) ou na CPU.

O vídeo processado sai por uma câmera virtual, então funciona em qualquer app: Discord, Meet, Zoom, Teams, OBS etc.

## Recursos

- Remoção de fundo com *video matting* (bordas suaves, com transparência gradual no cabelo) e consistência entre frames
- Fundo substituído por **cor sólida** (verde para chroma key por padrão), **imagem** ou **desfoque**
- Roda na GPU via DirectML (AMD, Intel, NVIDIA) ou na CPU
- Controles para equilibrar qualidade, FPS e consumo de GPU/CPU
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

### Diagnóstico

| Opção | Padrão | Descrição |
|---|---|---|
| `--profile` | desligado | Imprime a cada segundo o FPS e o tempo médio (ms) por etapa. |

Exemplo de saída do `--profile`:

```
24.0 FPS | leitura 17.5 ms | ia 13.0 ms | fluxo 0.0 ms | composicao 6.0 ms | envio 5.0 ms
```

- **leitura:** tempo esperando a webcam entregar o frame. Se for alto mesmo sem IA (`--mode none`), o limite é a câmera.
- **ia:** tempo da inferência. Com `--paralelo`, fica perto de zero e aparece `IA N/s`, que indica quantas vezes por segundo a IA rodou.
- **fluxo:** tempo da compensação de movimento (`--fluxo`).
- **composicao** e **envio:** mistura do fundo e envio para a câmera virtual.

## Atalhos no preview

Com a janela de preview aberta (`--preview`):

| Tecla | Ação |
|---|---|
| `b` | Fundo desfocado |
| `c` | Cor sólida |
| `i` | Imagem de fundo (só funciona se o programa foi iniciado com `--image`) |
| `n` | Sem efeito |
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

**FPS preso em 24 (ou 15).** Muitas webcams reduzem o FPS em ambientes escuros por causa da exposição automática. Aumente a luz, desative opções como "baixa luminosidade" ou "taxa de quadros dinâmica" no app da webcam e teste `--mode none --profile`: se o FPS continuar baixo sem IA, o limite é a câmera.

**Halo ou contorno em volta de mim.** Use `--shrink 2` e/ou `--edge 0.12`.

**A borda "arrasta" quando me mexo.** A máscara está velha. Use `--ia-fps` maior, adicione `--fluxo`, ou abandone `--skip` e `--paralelo`.

**Alto uso de GPU.** Reduza `--infer-width`, reduza `--fps`, use o modelo `mobilenet` (padrão) e evite `--skip`/`--ia-fps` baixos.

## Créditos

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), de Shanchuan Lin et al. (*Robust High-Resolution Video Matting with Temporal Guidance*, WACV 2022), licenciado sob GPL-3.0. O modelo não é distribuído neste repositório: é baixado do repositório original na primeira execução.
- [ONNX Runtime](https://onnxruntime.ai/) (DirectML), [OpenCV](https://opencv.org/), [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) e [OBS Studio](https://obsproject.com/).
