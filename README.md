# İleri Yapay Zekâ Laboratuvarı

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği "İleri Yapay Zekâ" dersindeki uygulamalı ödevlerim. Her ödevde güncel bir makalenin resmi kodunu Google Colab'da (T4 GPU) çalışır hâle getirdim, yöntemi kod üzerinden açıkladım, karşılaştığım hataları ve düzeltmeleri belgeledim.

> **English:** Lab assignments for an Advanced AI course: reproducing recent papers' official code on Colab, explaining each method through its code, comparing with baselines and documenting every fix needed to make the code run.

| Lab | Konu | Defter |
|---|---|---|
| 1 | **MemGraphRAG** (KDD'26): üç katmanlı bellekli Graph-RAG | [lab1-memgraphrag](lab1-memgraphrag/MemGraphRAG_WarmUp.ipynb) |
| 2 | **FedFST**: Federe graf sürekli öğrenme; POWER (KDD'25) ve FedSPA (CVPR'25) ile karşılaştırma | [lab2-fedfst](lab2-fedfst/FedFST_POWER_FedSPA.ipynb) |

---

## Lab 1: MemGraphRAG

Klasik GraphRAG her metin parçasından bilgiyi birbirinden bağımsız çıkarır. Bu yüzden ortaya çıkan graf tutarsız ve parçalı olabilir. MemGraphRAG çıkarıma ortak bir **üç katmanlı bellek** (şema ↔ olgu ↔ pasaj) ekler: olgular grafa yazılmadan önce bütüne bakılarak soyutlanır, süzülür ve çelişkileri giderilir.

- **LLM:** Qwen2.5-7B-Instruct (4-bit GGUF), Colab'da **Ollama** ile yerel olarak ve OpenAI uyumlu uç nokta üzerinden. API anahtarı gerekmez.
- **Embedding:** `BAAI/bge-large-en-v1.5`
- **Veri:** 2WikiMultiHopQA'dan küçük bir alt küme (çok adımlı sorular)
- **Adımlar:** indeksleme → retrieval + soru-cevap → EM/F1 değerlendirmesi → klasik RAG ile küçük karşılaştırma
- Defterin sonunda klasik RAG, GraphRAG/HippoRAG ve MemGraphRAG'ın indeksleme birimi, gürültü kontrolü, çelişki çözümü ve retrieval açısından karşılaştırması yer alıyor.

Kaynak kod: [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG)

## Lab 2: FedFST, POWER ve FedSPA

**Problem:** Federe graf sürekli öğrenme (FGCL). Graf, ham veri paylaşmayan 3 istemciye bölünür (Louvain). Sınıflar zaman içinde görevler hâlinde gelir, bu da felaket unutmaya yol açar. Metrikler: ortalama doğruluk **AA ↑** ve ortalama unutma **AF ↓**.

| Veri seti | Homofili | FedFST AA | FedFST AF | POWER AA | POWER AF | FedSPA test doğr.* |
|---|---|---|---|---|---|---|
| Cora | yüksek | **74,03** | 20,46 | 55,25 | **18,92** | 82,28 |
| CiteSeer | yüksek | **71,65** | **14,28** | 44,84 | 17,58 | 73,27 |
| Actor | düşük | **41,34** | 20,64 | 25,24 | **2,68** | 24,21 |
| Squirrel | düşük | **38,82** | 20,82 | 25,00 | **0,95** | 23,65 |
| Minesweeper | düşük | 49,53 | 43,80 | **76,69** | **0,00** | 83,01 |

\*FedSPA sürekli öğrenme yapmadığı için statik test doğruluğu raporlanmıştır. Tek tohum, % cinsinden.

**Bulgular**
- FedFST beş veri setinin dördünde POWER'dan belirgin biçimde yüksek AA veriyor (+13,8 ile +26,8 puan arası).
- Heterofilik graflarda (Actor, Squirrel) POWER şans seviyesinde (~%25) kalıyor. Düşük AF değeri iyi bir unutma direncinden değil, unutacak kadar öğrenmemiş olmasından kaynaklanıyor.
- Minesweeper'da her görev tek sınıf içerdiği için kurulum dejenere. POWER'ın avantajı büyük olasılıkla replay belleğinden geliyor.
- FedFST, sentetik graf optimizasyonu ve bilgi damıtma nedeniyle POWER'dan 1,5–4 kat yavaş.

**Orijinal kodlarda düzeltilen hatalar (seçme):** eksik `__init__.py` dosyalarının HuggingFace `datasets` paketiyle çakışması, tanımsız argümanlar, kurulamayan `torch_sparse`/DGL bağımlılıkları (`uv` ile ayrı Python 3.11 ortamı), PyTorch ≥ 2.6 `weights_only` değişikliği, Python döngüsüyle yazılmış kenar budamanın vektörleştirilmesi. Tam liste defterin 10. bölümünde.

## Lisans

Defterler ve açıklamalar MIT lisanslıdır. Bkz. [LICENSE](LICENSE). Çalıştırılan makale kodları kendi lisanslarına tabidir.
