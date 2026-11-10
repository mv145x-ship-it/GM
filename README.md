# Гравитационная модель российской внешней торговли

Репозиторий содержит данные, код и рукописи **двух статей**, подготовленных на основе выпускной квалификационной работы
(`ВКР.docx`, данные `Данные*.xlsx`, описание `Info.txt`):

| | Статья 1 | Статья 2 |
|---|---|---|
| Рукопись | `paper/article1_ru.docx` | `paper/article2_ru.docx` |
| Содержание | Логарифмически-линейная гравитационная модель по материалам ВКР: модели 1–8 (МНК, лаги, системный ОММ), расчёт ожидаемого оборота 2024 г., прогноз на 2025 г. и его сверка с фактом | Уточнённая (структурная) модель: PPML, эффекты «страна — год» и «пара», парные индикаторы санкций, event study, гравитационный ориентир, проверки вывода |
| Данные | исходный набор ВКР, 2000–2024 (`data/vkr_panel_renamed.csv`) | обновлённая панель, 2000–2025 (`data/gravity_panel_updated.csv`) |
| Шаблон текста | `paper/parts_a1/*.md` | `paper/parts_a2/*.md` |
| Сборка | `python code/12_build_article1.py` | `python code/08_build_paper.py` |

## Данные

* `data/vkr_panel_renamed.csv|xlsx` — исходные данные ВКР (496 пар × 2000–2024) с переименованными по общепринятой нотации переменными.
* `data/gravity_panel.csv|xlsx` — гармонизированная панель на данных ВКР (2000–2024): расстояния CEPII, ВВП и население WDI,
  исправленные дамми членства, парные индикаторы санкций.
* `data/gravity_panel_updated.csv` — **основная панель статьи 2** (2000–2025, 12 890 наблюдений). Получена скриптом
  `code/10_merge_update.py` из `gravity_panel.csv` и свежих выгрузок (`data/raw_update/`, ноябрь 2026 г.):
  * торговля 2019–2024 гг. сверена с сообщениями обеих стран пары в UN Comtrade: из 2971 значения подтверждено 2519, заменено 452
    (ошибки исходного набора сосредоточены в 2022–2024 гг.: пары с ОАЭ, Канадой, Финляндией);
  * добавлен 2025 г. (490 пар из 496; нет отчётности Беларуси, ОАЭ и Украины — в т. ч. нет пар Россия — Беларусь, Россия — ОАЭ, Россия — Украина);
  * для пар с Россией используются сообщения партнёров (Россия не публикует детальную статистику с 2022 г.);
  * ВВП и население — WDI по 2025 г. (ОАЭ, 2025 — оценка МВФ WEO);
  * **значения торговли за 2000–2018 гг. не сверялись**: суточная квота открытой точки UN Comtrade была исчерпана
    (в кэше — сообщения 8 стран за эти годы). Досверить можно командами из раздела «Обновление данных».
* Словарь переменных — `data/DATA_DICTIONARY.md`; отчёт об аудите исходного набора — `data/data_quality_report.csv`;
  отчёт об обновлении — `data/update_report.csv`.

Обозначения: `trade` — суммарная торговля пары (экспорт + импорт), млн долл. США, текущие цены; `gdp_i`, `gdp_j` — ВВП, млрд долл.;
`dist` — население-взвешенное расстояние CEPII (км); суффиксы `_i`, `_j` — страны пары (`iso3_i < iso3_j`), `_ij` — парные переменные.
Санкционные индикаторы парные: `sanc14_ij` = «Россия × ядро коалиции 2014 г.» × 1[t ≥ 2014], `sanc22_ij` — то же для t ≥ 2022.

## Код

```
code/gravity_common.py     страны, группы, хронология членства (ЕС, НАТО, БРИКС, ЕАЭС, ВТО)
code/01_build_dataset.py   сборка и аудит наборов данных из файлов ВКР
code/02_replicate_vkr.py   модели 1–7 ВКР (МНК) — точное воспроизведение вывода gretl
code/11_article1_results.py  статья 1: таблицы, прогнозы 2024/2025, рисунки
code/12_build_article1.py  статья 1: сборка рукописи
code/ppml.py               собственная реализация PPML с разреженными ФЭ «страна — год» и «пара»
code/03_validate_estimator.py  проверка ppml.py на pyfixest
code/04_main_models.py     статья 2: таблицы 1–4, подгруппы, партнёрские сдвиги, leave-one-out, тесты Вальда
code/05_event_study_counterfactual.py  статья 2: event study и гравитационный ориентир
code/06_inference_checks.py  статья 2: перестановочный тест, бутстрэп, диадно-робастные ошибки
code/07_figures.py         статья 2: рисунки
code/08_build_paper.py     статья 2: сборка рукописи;  code/09_make_docx.py — Markdown → DOCX (pandoc)
code/fetch_update.py       загрузка данных из API (WDI, IMF WEO, UN Comtrade)
code/10_merge_update.py    сверка и объединение выгрузок с панелью → data/gravity_panel_updated.csv
gretl/01_vkr_replication.inp, 03_vkr_dynamic_panel.inp   статья 1: модели ВКР в gretl (модель 8 — системный ОММ)
gretl/02_ppml_gravity.inp  статья 2: PPML в gretl (на панели 2000–2024)
tests/test_update_pipeline.py  тесты конвейера обновления данных
```

## Воспроизведение

```bash
pip install -r requirements.txt            # + pandoc (для DOCX); gretl — для скриптов gretl/

# статья 1
python code/02_replicate_vkr.py
gretlcli -b /полный/путь/gretl/03_vkr_dynamic_panel.inp     # модель 8 → results/vkr_model8_dpanel.csv (в скрипте указать set workdir)
python code/11_article1_results.py && python code/12_build_article1.py

# статья 2
python code/04_main_models.py
python code/05_event_study_counterfactual.py
python code/06_inference_checks.py --perm 500 --boot 200    # самый долгий шаг
python code/07_figures.py
python code/08_build_paper.py
```

Готовые `data/*.csv` и `results/*` лежат в репозитории, поэтому рукописи можно пересобрать без пересчёта.
Все числа в текстах и таблицах подставляются из `results/`; интерпретационные формулировки (`paper/parts_a*/`) при смене данных
нужно перечитывать вручную. Шаг 01 требует клонов `github.com/datasets/gdp`, `github.com/datasets/population` и
`github.com/pachadotdev/cepiigeodist` (пути — в аргументах скрипта).

## Обновление данных

```bash
python code/fetch_update.py probe                               # доступность хостов
python code/fetch_update.py wdi --start 2000 --end 2025
python code/fetch_update.py weo --start 2000 --end 2026
# UN Comtrade без ключа (открытая точка: один год на запрос, суточная квота ~ несколько сотен запросов; кэш в data/raw_update/cache):
python code/fetch_update.py comtrade-public --start 2000 --end 2025 [--supplement]
# либо с ключом (comtradeplus.un.org), несколько лет на запрос:
COMTRADE_API_KEY=... python code/fetch_update.py comtrade --start 2000 --end 2025
python code/10_merge_update.py --end-year 2025 --replace verify --partner-only RUS
python tests/test_update_pipeline.py
```

Политики замены (`--replace none|flagged|verify|all`) описаны в `code/10_merge_update.py`. Исходная панель задаётся
переменной окружения `GM_PANEL` (по умолчанию — `data/gravity_panel_updated.csv`).

## Что нужно сделать автору перед отправкой

* Заполнить плейсхолдеры: `[Фамилия И. О.]`, сведения об организации, «Благодарности».
* **Сверить списки литературы.** Часть выходных данных сверена по открытым библиографическим записям, остальные составлены по памяти;
  записи Смирнова (2020) и Упатиной (2021) взяты из ВКР и не проверялись; реквизиты нормативных актов (распоряжение № 430-р, указ № 560)
  следует проверить по официальным источникам.
* В статье 1 «санкционные» дамми ВКР названы индикаторами периодов (они равны единице для всех пар) и не трактуются как эффект санкций;
  сравнение прогноза на 2025 г. с фактом добавлено сверх содержания ВКР.
