import random, json, argparse
from tqdm import tqdm

# Расширенные списки данных
brands = ["coca cola", "lays", "danone", "prostokvashino", "pepsi", "nestle", "heinz", "lipton", "milka", "mars", "Агуша", "Простоквашино", "Слобода", "Добрый", "Макфа", "Горячая штучка", "Бабаевский", "Зеленый Дубок", "Рыбное Выбор", "Я", "ФрутоНяня", "Махеев", "Красная цена", "Дарья", "Черкизово", "Останкино", "Мираторг", "Кухмастер", "Красный Октябрь", "Алёнка", "Ягода", "Адыл", "Балтимор", "Вкуснотеево", "Вкусно", "Добрые Печеньки", "Едадил", "Жито", "Завтра", "Интерторг", "Каждый День", "Красный пищевик", "Кукусик", "Лакомка", "Лебедянский", "Лидский", "Лимби", "Макарошки", "Макфа", "Мандарин", "Моршинская", "Морозко", "Националь", "Нестле", "Орео", "Папа может", "Пионер", "Пончики", "Просто", "Роллтон", "Россия", "Русская Каса", "Русский хлеб", "Свитлогорье", "Сибирская", "Солнечные", "Сочные", "Столица", "Тайна", "Тирамису", "Торговая группа Союз", "Торговый дом Ах", "Тульчинка", "Умка", "Усадьба", "Фасоль", "Филимон", "Флорида", "Фреш", "Хороший", "Хрустим", "Чайный дом", "Черная жемчужина", "Чипсы", "Шоколадка", "Штольц", "Щедрая осень", "Элемент", "Эликсир", "Эльдорадо", "Энергия", "Этно", "Южный", "Яблочная", "Ямал", "Ясная поляна"]

types = ["молоко", "хлеб", "чипсы", "сок", "вода", "сыр", "йогурт", "шоколад", "пиво", "печенье", "молоко", "кефир", "ряженка", "сметана", "сливки", "йогурт", "творог", "сыр", "масло сливочное", "масло растительное",
    "хлеб", "батон", "багет", "лаваш", "лепешка", "булочка", "сухари", "крекер", "печенье", "пряники",
    "яйца", "курица", "филе куриное", "бедро куриное", "голень куриная", "индейка", "свинина", "говядина", "фарш", "сосиски",
    "рыба", "семга", "форель", "треска", "минтай", "сельдь", "икра", "крабовые палочки", "морская капуста", "консервы рыбные",
    "картофель", "морковь", "лук", "чеснок", "помидоры", "огурцы", "капуста", "свекла", "кабачок", "баклажан",
    "яблоки", "бананы", "груши", "апельсины", "мандарины", "лимоны", "виноград", "сливы", "персики", "гранат",
    "рис", "гречка", "овсянка", "пшено", "перловка", "макароны", "вермишель", "лапша", "мука", "сахар",
    "соль", "перец", "специи", "кетчуп", "майонез", "соевый соус", "горчица", "хрен", "аджика", "соус барбекю",
    "шоколад", "конфеты", "карамель", "пастила", "зефир", "мармелад", "вафли", "торт", "пирожное", "мороженое",
    "сок", "вода", "минералка", "газировка", "чай", "кофе", "какао", "компот", "квас", "энергетик"
]

# Расширенные единицы измерения для VOLUME
volume_numbers = ["0.33", "0.5", "1", "1.5", "2", "2.5", "3", "5", "10", "15", "20", "25", "30", "50", "100", "150", "200", "250", "300", "330", "400", "500", "600", "750", "900", "1000"]
volume_units = ["л", "мл", "г", "кг", "шт", "уп", "пакет", "банка", "бутылка", "упаковка"]

# Расширенные проценты для PERCENT
perc_numbers = ["0.5", "1", "1.5", "2", "2.5", "3", "3.2", "4", "5", "6", "7", "8", "9", "10", "12", "15", "18", "20", "25", "30", "35", "40", "45", "50", "60", "70", "75", "80", "90", "100"]
perc_units = ["%", "процентов", "percent"]

# Составные процентные выражения для I-PERCENT
percent_phrases = [
    ["жирности", "5", "%"], ["жирности", "10", "%"], ["жирности", "15", "%"],
    ["сахара", "5", "%"], ["сахара", "10", "%"], ["сахара", "15", "%"],
    ["спирта", "5", "%"], ["спирта", "10", "%"], ["спирта", "15", "%"],
    ["какао", "30", "%"], ["какао", "50", "%"], ["какао", "70", "%"],
    ["содержание", "белка", "20", "%"], ["содержание", "белка", "25", "%"],
    ["концентрация", "10", "%"], ["концентрация", "20", "%"]
]

# Составные объемные выражения для I-VOLUME
volume_phrases = [
    ["объем", "1", "л"], ["объем", "0.5", "л"], ["объем", "2", "л"],
    ["вес", "100", "г"], ["вес", "200", "г"], ["вес", "500", "г"],
    ["масса", "1", "кг"], ["масса", "2", "кг"], ["масса", "5", "кг"],
    ["в", "упаковке", "10", "шт"], ["в", "упаковке", "20", "шт"],
    ["пачка", "250", "г"], ["пачка", "500", "г"],
    ["бутылка", "1", "л"], ["бутылка", "1.5", "л"]
]

# Филлеры
fillers = [
    ["без", "сахара"], ["без", "лактозы"], ["натуральный"], ["для", "детей"], ["без", "глютена"],
    ["низкая", "цена"], ["со", "вкусом", "шоколада"], ["с", "бананом"], ["органический"], ["премиум"],
    ["классический"], ["деревенский"], ["домашний"], ["свежий"], ["охлажденный"], ["замороженный"]
]

# Вероятности
P_BRAND = 0.7
P_TYPE = 0.85
P_VOLUME = 0.6
P_PERCENT = 0.3
P_FILLER = 0.15
P_COMPLEX_PERCENT = 0.4  # вероятность составного процента
P_COMPLEX_VOLUME = 0.3   # вероятность составного объема

# Функция опечаток (используется только если enable_typos=True)
def typo_token(tok, prob=0.18):
    if random.random() > prob or len(tok) < 2:
        return tok
    i = random.randrange(len(tok))
    op = random.choice(["drop", "swap", "dup", "replace"])
    if op == "drop":
        return tok[:i] + tok[i+1:]
    if op == "swap" and i < len(tok)-1:
        return tok[:i] + tok[i+1] + tok[i] + tok[i+2:]
    if op == "dup":
        return tok[:i] + tok[i] + tok[i] + tok[i+1:]
    if op == "replace":
        letters = "абвгдеёжзийклмнопрстуфхцчшщьыэюяabcdefghijklmnopqrstuvwxyz"
        return tok[:i] + random.choice(letters) + tok[i+1:]
    return tok

def maybe_typos(tokens, prob=0.18):
    return [typo_token(t, prob) for t in tokens]

def add_entity(tokens, labels, entity_text, label_prefix, apply_typos=False):
    ent_tokens = entity_text.split()
    
    if apply_typos:
        ent_tokens = maybe_typos(ent_tokens)
    
    for i, t in enumerate(ent_tokens):
        tokens.append(t)
        if i == 0:
            labels.append(f"B-{label_prefix}")
        else:
            labels.append(f"I-{label_prefix}")

def add_complex_percent(tokens, labels, apply_typos=False):
    phrase = random.choice(percent_phrases)
    
    if apply_typos:
        phrase = maybe_typos(phrase)
    
    for i, token in enumerate(phrase):
        tokens.append(token)
        if token in perc_numbers or token == "%":
            if i > 0 and labels[-1].startswith("B-PERCENT"):
                labels.append("I-PERCENT")
            else:
                labels.append("B-PERCENT")
        else:
            labels.append("I-PERCENT")

def add_complex_volume(tokens, labels, apply_typos=False):
    phrase = random.choice(volume_phrases)
    
    if apply_typos:
        phrase = maybe_typos(phrase)
    
    for i, token in enumerate(phrase):
        tokens.append(token)
        if token in volume_numbers or token in volume_units:
            if i > 0 and labels[-1].startswith("B-VOLUME"):
                labels.append("I-VOLUME")
            else:
                labels.append("B-VOLUME")
        else:
            labels.append("I-VOLUME")

def add_simple_percent(tokens, labels, apply_typos=False):
    pnum = random.choice(perc_numbers)
    unit = random.choice(perc_units)
    
    if apply_typos:
        pnum = typo_token(pnum)
        unit = typo_token(unit)
    
    tokens.append(pnum)
    labels.append("B-PERCENT")
    tokens.append(unit)
    labels.append("I-PERCENT")

def add_simple_volume(tokens, labels, apply_typos=False):
    num = random.choice(volume_numbers)
    unit = random.choice(volume_units)
    
    if apply_typos:
        num = typo_token(num)
        unit = typo_token(unit)
    
    tokens.append(num)
    labels.append("B-VOLUME")
    tokens.append(unit)
    labels.append("I-VOLUME")

def tokens_to_char_spans(tokens, labels):
    sample = " ".join(tokens)
    spans = []
    cursor = 0
    for tok, lab in zip(tokens, labels):
        start = cursor
        end = start + len(tok)
        spans.append((start, end, lab))
        cursor = end + 1  # учёт пробела
    return sample, spans

def build_one_sample(enable_typos=False):
    tokens = []
    labels = []

    # BRAND
    if random.random() < P_BRAND:
        add_entity(tokens, labels, random.choice(brands), "BRAND", apply_typos=enable_typos)

    # FILLER между brand и type
    if random.random() < P_FILLER:
        f = random.choice(fillers)
        for tok in f:
            tokens.append(tok)
            labels.append("O")

    # TYPE
    if random.random() < P_TYPE:
        add_entity(tokens, labels, random.choice(types), "TYPE", apply_typos=enable_typos)

    # VOLUME
    if random.random() < P_VOLUME:
        if random.random() < P_COMPLEX_VOLUME:
            add_complex_volume(tokens, labels, apply_typos=enable_typos)
        else:
            add_simple_volume(tokens, labels, apply_typos=enable_typos)

    # PERCENT
    if random.random() < P_PERCENT:
        if random.random() < P_COMPLEX_PERCENT:
            add_complex_percent(tokens, labels, apply_typos=enable_typos)
        else:
            add_simple_percent(tokens, labels, apply_typos=enable_typos)

    # Если ничего не добавлено
    if len(tokens) == 0:
        tokens.append("продукт")
        labels.append("O")

    # Случайная перестановка
    if random.random() < 0.06 and len(tokens) > 2:
        k = random.randint(1, len(tokens)-1)
        tokens = tokens[k:] + tokens[:k]
        labels = labels[k:] + labels[:k]

    sample, spans = tokens_to_char_spans(tokens, labels)
    return {"sample": sample, "annotation": spans}

def generate(output_path, n=50000, dedupe=True, enable_typos=False):
    seen = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for _ in tqdm(range(n * 3), desc="Generating samples"):
            item = build_one_sample(enable_typos=enable_typos)
            s = item["sample"]
            if dedupe and s in seen:
                continue
            seen.add(s)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count >= n:
                break
    print(f"Saved {count} samples to {output_path}")
    print(f"Oпечатки: {'ВКЛЮЧЕНЫ' if enable_typos else 'ВЫКЛЮЧЕНЫ'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--typos", action="store_true", help="Включить генерацию опечаток")
    args = parser.parse_args()
    generate(args.output, n=args.n, enable_typos=args.typos)