# AB Reader

Lettore locale di ebook EPUB con voce sintetica **Kokoro**: legge ad alta voce
evidenziando la parola in corso, ricorda dove sei arrivato e tiene gli appunti
di fianco al testo. Gira interamente sul tuo computer, senza rete e senza account.

## Installazione

Serve **Python 3.10 o superiore**. Poi, una volta sola:

    setup.bat

Crea l'ambiente virtuale, installa le dipendenze e scarica i modelli vocali
(circa 350 MB, che finiscono in `data/models/`).

## Avvio

    run.bat

Si apre una finestra desktop. Per aprirlo invece nel browser:

    run.bat --browser

### Se non parte

`run.bat` usa `pythonw.exe`, che non ha una console: tutto ciò che l'applicazione
stampa finisce in **`data/abreader.log`**. Se la finestra non compare, è lì che
guardare. Un avvio riuscito scrive tre righe (server pronto, modello caricato,
motore pronto). Per vedere gli errori a schermo mentre succedono:

    .venv\Scripts\python.exe app.py

## Come si usa

1. **Aggiungi un libro**: pulsante *Aggiungi un EPUB*, oppure trascina il file
   nella finestra.
2. **Premi play**. La frase in lettura ha uno sfondo tenue, la parola in corso
   è evidenziata.
3. **Clic su una parola**: la lettura riparte esattamente da lì.
4. **Appunti**: il pannello di destra. Un appunto si aggancia alla frase che
   stai ascoltando; se prima selezioni del testo, si aggancia alla selezione.
   Cliccando la citazione di un appunto torni a quel punto del libro.
5. La posizione si salva da sola. *Riprendi* torna all'ultimo punto salvato,
   anche da un altro capitolo.

### Tastiera

| Tasto | Azione |
|---|---|
| `spazio` | play / pausa |
| `→` | frase successiva |
| `←` | frase precedente (o inizio di quella corrente) |
| `n` | nuovo appunto sulla frase corrente |

## Come funziona

    EPUB ──▶ capitoli ──▶ frasi ──▶ parole        (abreader/epub_parser.py, textseg.py)
                             │
                             ▼
                    Kokoro ONNX ──▶ audio + durata di ogni fonema
                             │                     (abreader/tts/engine.py)
                             ▼
                  tempi di inizio/fine di ogni parola
                             │
                             ▼
            Web Audio API: ogni frase è un buffer schedulato
            a un istante assoluto → la parola da evidenziare
            è una sottrazione, non una stima      (web/app.js)

Il punto che rende l'evidenziazione precisa è l'uscita `duration` del modello
Kokoro (release **model-files-v1.1**): dà la durata di ogni fonema, e i fonemi
sono separati da spazi esattamente dove lo sono le parole nel testo. La
corrispondenza è uno-a-uno sia in italiano sia in inglese; nei casi rari in cui
non torna (numeri scritti in cifre, abbreviazioni espanse) si ripiega su una
ripartizione proporzionale, limitata alla singola frase.

Le frasi sono tenute sotto i 200 caratteri: accorcia l'attesa iniziale e impedisce
alla deriva di allineamento di accumularsi.

Le frasi brevi (fino a 4 parole: "Sì.", "Non lo so.") da sole suonano
artificiose, perché il modello non ha abbastanza contesto per l'intonazione.
Vengono quindi sintetizzate insieme alla precedente e alla successiva, sempre
entro i 200 caratteri, e l'audio viene poi tagliato nella pausa fra l'una e
l'altra. Per il lettore restano frasi separate, con la loro evidenziazione.

## Pronuncia

I fonemi passati a Kokoro sono preparati come nella pipeline ufficiale del
modello (`abreader/tts/g2p.py`): è la forma su cui le voci sono state
addestrate. In particolare *ci/gi/z/zz* arrivano come un suono solo, e le
virgolette « » restano, così le battute di dialogo hanno la loro intonazione.

Nei libri in italiano le **parole inglesi** vengono riconosciute e lette con la
pronuncia inglese (britannica): *weekend*, *computer*, *cool*, e le frasi
inglesi intere come "Oh my God". Il riconoscimento usa un elenco di anglicismi
(`abreader/tts/anglicismi.txt`) e alcune regole sulla grafia, prudenti: nel
dubbio la parola resta italiana.

Per correggere una pronuncia c'è **`data/pronunce.txt`**, creato al primo avvio:

    cool = en            leggila in inglese
    yogurt = it          leggila in italiano
    Hermione = ermaiòni  leggila come se fosse scritta così
    apple pie = en       anche più parole insieme (solo en o it)

Le parole composte col trattino possono essere miste: in *cyber-futurismo*
la prima metà si legge in inglese, la seconda in italiano.

Le correzioni valgono subito per le frasi nuove; le frasi in cui la pronuncia
cambia vengono risintetizzate da sole, il resto della cache resta valido.

### Prestazioni

Su CPU senza GPU la sintesi va a circa **1,3× il tempo reale**: la prima frase si
fa attendere qualche secondo, poi il lettore sintetizza in anticipo (4 frasi) e
la lettura scorre continua. L'audio prodotto resta in cache su disco
(`data/cache/tts/`), quindi rileggere o tornare indietro è istantaneo.

Cambiare **voce** o **velocità** invalida la cache: sono parametri della sintesi,
non della riproduzione, e l'audio va rifatto.

## Voci

54 voci. Le predefinite sono `if_sara` per l'italiano e `af_heart` per l'inglese;
la lingua del libro viene rilevata dai metadati dell'EPUB, con verifica sul testo.
Il menu a tendina propone per prime le voci della lingua del libro, ma restano
tutte selezionabili.

## Struttura

    abreader/
      config.py           percorsi e impostazioni
      textseg.py          divisione in frasi e parole (IT/EN)
      epub_parser.py      EPUB -> capitoli -> frasi, con cache JSON
      store.py            SQLite: libreria, posizione, appunti
      server.py           API HTTP
      download_models.py  primo scaricamento dei pesi
      tts/
        engine.py         Kokoro + allineamento parola-per-parola
        g2p.py            testo -> fonemi, con le parole inglesi
        anglicismi.txt    elenco delle parole inglesi riconosciute
        cache.py          cache audio su disco e coda di sintesi
    web/                  interfaccia (HTML/CSS/JS, nessun framework)
    app.py                avvio: server + finestra desktop
    data/                 modelli, libri, cache, database (non versionata)

## Dati

Tutto sta in `data/`. Per ripartire da zero basta cancellarla (si perdono
appunti e posizioni). Per liberare solo lo spazio audio, cancella
`data/cache/tts/`.

## Limiti attuali

- Solo EPUB (niente PDF o MOBI).
- Immagini e tabelle non vengono mostrate: si legge il testo.
- Nessuna analisi del sentiment: le inflessioni sono quelle che Kokoro ricava
  dalla punteggiatura. È il passo successivo previsto.

## Licenza

Codice rilasciato sotto licenza [MIT](LICENSE). Le dipendenze installate da
`requirements.txt` hanno licenze proprie: in particolare `phonemizer` ed
`espeak-ng` sono GPL-3.0, e i pesi del modello Kokoro sono Apache-2.0.
