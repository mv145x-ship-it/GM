---
title: "Гравитационная модель внешней торговли России: оценки на панели 32 стран за 2000–2024 гг. и опыт краткосрочного прогнозирования"
author:
  - "[Фамилия И. О.]"
  - "[Должность, организация, e-mail]"
---

::: {custom-style="Abstract"}
**Аннотация.** В статье оцениваются факторы двустороннего товарооборота России и её основных торговых партнёров на основе гравитационной модели в логарифмически-линейной форме. Эмпирическая база — сбалансированная панель из 496 пар 32 стран за 2000–2024 гг. Последовательно оцениваются статические спецификации (пулированный МНК) с ВВП на душу населения, расстоянием, общей границей и индикаторами членства в ЕС, БРИКС и НАТО, а затем динамические спецификации с лагами зависимой переменной, включая системный обобщённый метод моментов. Статические модели объясняют около четверти вариации товарооборота (R² от <<v.fit[1].r2>> до <<v.fit[4].r2>>) и характеризуются сильной автокорреляцией остатков; включение лага повышает R² до <<v.fit[5].r2>>, а коэффициент при лаге (<<v.m[5].l_trade_lag1.s>> в МНК и <<v.m[8].l_trade_lag1.s>> в системном ОММ) указывает на высокую инерционность торговых потоков. В динамических моделях средний уровень товарооборота в период с 2014 г. ниже на 9–16 % при прочих равных, а совместное членство в БРИКС связано с более высоким товарооборотом. На основе динамических моделей выполнены два расчёта: ожидаемый оборот России с партнёрами в 2024 г. по соотношениям, сложившимся до 2014 г., и прогноз на 2025 г.<% if v.f25.has_actual %> Сопоставление прогноза с фактическими данными за 2025 г. показывает, что модель с высокой инерцией не превосходит «наивный» прогноз неизменного оборота.<% endif %> Обсуждаются ограничения спецификации, прежде всего невозможность отделить эффект санкций от общих для всех стран изменений при использовании индикаторов периода.

**Ключевые слова:** гравитационная модель; внешняя торговля России; панельные данные; динамическая панельная модель; системный ОММ; прогнозирование товарооборота; gretl.

**JEL:** F14, F17, C23, C53.
:::

::: {custom-style="Abstract"}
**Abstract.** The paper estimates the determinants of bilateral trade between Russia and its main trading partners using a log-linear gravity model. The data are a balanced panel of 496 country pairs from 32 countries for 2000–2024. We estimate static specifications (pooled OLS) with GDP per capita, distance, contiguity and EU, BRICS and NATO membership indicators, and then dynamic specifications with lags of the dependent variable, including system GMM. The static models explain about a quarter of the variation in trade (R² from <<v.fit[1].r2|en>> to <<v.fit[4].r2|en>>) and exhibit strong residual autocorrelation; adding the lag raises R² to <<v.fit[5].r2|en>>, and the lag coefficient (<<v.m[5].l_trade_lag1.s|en>> in OLS and <<v.m[8].l_trade_lag1.s|en>> in system GMM) points to strong persistence of trade flows. In the dynamic models the average level of trade since 2014 is 9–16% lower, other things equal, and joint BRICS membership is associated with higher trade. Two calculations are based on the dynamic models: Russia's expected trade with partners in 2024 under pre-2014 relationships, and a forecast for 2025.<% if v.f25.has_actual %> A comparison of the forecast with actual 2025 data shows that the highly persistent model does not outperform a naive no-change forecast.<% endif %> We discuss the limitations of the specification, above all the impossibility of separating the effect of sanctions from changes common to all countries when period indicators are used.

**Keywords:** gravity model; Russian foreign trade; panel data; dynamic panel model; system GMM; trade forecasting; gretl.
:::
