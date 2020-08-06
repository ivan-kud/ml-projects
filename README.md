# My DataScience Projects

Here you can see my projects of DataScience.

## Exploratory data analysis
**Project Description:** Вас пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру.

**Goal:** Отследить влияние условий жизни учеников школы в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять учеников, находящихся в группе риска.

**Objectives:** Построить модель, которая предсказывала бы результат госэкзамена по математике для каждого ученика.

**Dataset Description:** 

Dataset features:
1. school — аббревиатура школы, в которой учится ученик;
2. sex — пол ученика ('M' - мужской, 'F' - женский);
3. age — возраст ученика (от 15 до 22 лет);
4. address — место проживания ученика ('U' - в городе, 'R' - за городом);
5. famsize — размер семьи('LE3' - до 3 человек, 'GT3' - более трех человек);
6. Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - живут раздельно);
7. Medu — образование матери (0 - нет, 1 - 4 класса, 2 - от 5 до 9 классов, 3 - среднее специальное или 11 классов, 4 - высшее);
8. Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - от 5 до 9 классов, 3 - среднее специальное или 11 классов, 4 - высшее);
9. Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос. служба, 'at_home' - не работает, 'other' - другое);
10. Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос. служба, 'at_home' - не работает, 'other' - другое);
11. reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое);
12. guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое);
13. traveltime — время в пути до школы (1 - менее 15 мин, 2 - 15-30 мин, 3 - 30-60 мин, 4 - более 60 мин);
14. studytime — время на учёбу помимо школы в неделю (1 - менее 2 ч, 2 - 2-5 ч, 3 - 5-10 ч, 4 - более 10 ч);
15. failures — количество внеучебных неудач (n, если 0 <= n <= 3, иначе 4);
16. schoolsup — дополнительная образовательная поддержка школы (yes или no);
17. famsup — семейная образовательная поддержка (yes или no);
18. paid — дополнительные платные занятия по математике (yes или no);
19. activities — дополнительные внеучебные занятия (yes или no);
20. nursery — посещал детский сад (yes или no);
21. Gstudytime — studytime, granular;
22. higher — желание получить высшее образование (yes или no);
23. internet — наличие интернета дома (yes или no);
24. romantic — в романтических отношениях (yes или no);
25. famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо);
26. freetime — свободное время после школы (от 1 - очень мало до 5 - очень много);
27. goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много);
28. health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо);
29. absences — количество пропущенных занятий;
30. score — баллы по госэкзамену по математике.

## About tasty and healthy food
**Project Description:** TripAdvisor rating prediction of european restaurants.

**Goal:** To identify restaurants that inflate their ratings.

**Objectives:** Create model to predict restaurant rating.

**Dataset Description:** Dataset consists of 40 000 rows into train set and 10 000 rows into test set.

Dataset features:
1. Restaurant_id — identification number of restaurant / restaurant network;
2. City — city where the restaurant is located;
3. Cuisine Style — cuisine or cuisines, which include dishes offered in the restaurant;
4. Ranking — the spot that this restaurant takes among all the restaurants in its city;
5. Rating — restaurant rating according to TripAdvisor (target variable);
6. Price Range — restaurant price range;
7. Number of Reviews — restaurant reviews amount;
8. Reviews — data of two reviews displayed on the restaurant website;
9. URL_TA — restaurant URL page on TripAdvosor;
10. ID_TA — restaurant identifier in TripAdvisor database.
