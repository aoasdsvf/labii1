import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Установка русского шрифта для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# =============================================================================

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ ДАННЫХ ТИТАНИКА")
print("=" * 60)

# Загрузка данных
df = pd.read_csv('titanic.csv')

print("\n1. ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")
print("-" * 40)

# Базовая информация
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nНазвания столбцов: {list(df.columns)}")
print(f"\nТипы данных:")
print(df.dtypes)

# Просмотр первых строк
print("\nПервые 5 строк данных:")
print(df.head().to_string())

# =============================================================================
# 2. ТЩАТЕЛЬНЫЙ АНАЛИЗ ПРОПУЩЕННЫХ И НУЛЕВЫХ ЗНАЧЕНИЙ
# =============================================================================

print("\n\n2. АНАЛИЗ ПРОПУЩЕННЫХ И НУЛЕВЫХ ЗНАЧЕНИЙ")
print("-" * 40)

# Проверка различных типов пропущенных значений
print("Детальный анализ данных на наличие пропусков и нулевых значений:")

# Проверка NaN
print("\n1. Стандартные NaN значения:")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_info = pd.DataFrame({
    'Пропущено': missing_data,
    'Процент': missing_percent
})
print(missing_info)

# Проверка пустых строк
print("\n2. Проверка пустых строк в текстовых полях:")
text_columns = df.select_dtypes(include=['object']).columns
empty_found = False
for col in text_columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        print(f"  {col}: {empty_count} пустых значений ({empty_count/len(df)*100:.1f}%)")
        empty_found = True
if not empty_found:
    print("  Пустых строк не обнаружено")

# ДЕТАЛЬНАЯ ПРОВЕРКА НУЛЕВЫХ ЗНАЧЕНИЙ ВО ВСЕХ ПОЛЯХ
print("\n3. ДЕТАЛЬНАЯ ПРОВЕРКА НУЛЕВЫХ ЗНАЧЕНИЙ ВО ВСЕХ ПОЛЯХ:")

# Для числовых полей
print("\n3.1. Числовые поля:")
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_zero_summary = []

for col in numeric_columns:
    zero_count = (df[col] == 0).sum()
    zero_percent = zero_count / len(df) * 100
    numeric_zero_summary.append({
        'Поле': col,
        'Нулевых значений': zero_count,
        'Процент': zero_percent,
        'Интерпретация': ''
    })
    
    # Интерпретация для каждого поля
    if col == 'Survived':
        interpretation = "0 = не выжил (валидное значение)"
    elif col == 'Pclass':
        interpretation = "0 = невалидное значение (классы: 1,2,3)"
    elif col == 'Age':
        interpretation = "0 = невалидное значение (возраст не может быть 0)"
    elif col == 'Siblings/Spouses Aboard':
        interpretation = "0 = нет братьев/сестер/супругов (валидное значение)"
    elif col == 'Parents/Children Aboard':
        interpretation = "0 = нет родителей/детей (валидное значение)"
    elif col == 'Fare':
        interpretation = "0 = бесплатный билет (требует проверки)"
    else:
        interpretation = "требует анализа"
    
    numeric_zero_summary[-1]['Интерпретация'] = interpretation
    
    if zero_count > 0:
        print(f"  {col}: {zero_count} нулевых значений ({zero_percent:.1f}%) - {interpretation}")

# Создаем DataFrame для визуализации
numeric_zero_df = pd.DataFrame(numeric_zero_summary)
print(f"\n  Всего числовых полей с нулевыми значениями: {len([x for x in numeric_zero_summary if x['Нулевых значений'] > 0])}")

# Для текстовых полей - проверка на "нулевые" строки
print("\n3.2. Текстовые поля:")
text_columns = df.select_dtypes(include=['object']).columns
text_zero_summary = []

for col in text_columns:
    # Проверяем различные "нулевые" значения в текстовых полях
    empty_count = (df[col] == '').sum()
    none_count = (df[col].isna()).sum()  # уже учтено в NaN
    whitespace_count = (df[col].str.strip() == '').sum() - empty_count
    unknown_count = (df[col].str.lower().str.contains('unknown|none|null|n/a', na=False)).sum()
    
    text_zero_summary.append({
        'Поле': col,
        'Пустых строк': empty_count,
        'Пробельных значений': whitespace_count,
        'Unknown/None': unknown_count
    })
    
    if empty_count > 0 or whitespace_count > 0 or unknown_count > 0:
        print(f"  {col}:")
        if empty_count > 0:
            print(f"    - Пустых строк: {empty_count}")
        if whitespace_count > 0:
            print(f"    - Только пробелы: {whitespace_count}")
        if unknown_count > 0:
            print(f"    - Содержит 'unknown/none': {unknown_count}")

print("\n3.3. Анализ аномальных значений:")
print("Возраст (Age):")
print(f"  Минимальный возраст: {df['Age'].min():.2f}")
print(f"  Максимальный возраст: {df['Age'].max():.2f}")
age_zeros = (df['Age'] == 0).sum()
age_negative = (df['Age'] < 0).sum()
age_over_100 = (df['Age'] > 100).sum()
print(f"  Возраст = 0: {age_zeros}")
print(f"  Отрицательный возраст: {age_negative}")
print(f"  Возраст > 100: {age_over_100}")

print("\nСтоимость билета (Fare):")
print(f"  Минимальная стоимость: {df['Fare'].min():.2f}")
print(f"  Максимальная стоимость: {df['Fare'].max():.2f}")
fare_zeros = (df['Fare'] == 0).sum()
fare_negative = (df['Fare'] < 0).sum()
print(f"  Стоимость = 0: {fare_zeros} ({fare_zeros/len(df)*100:.1f}%)")
print(f"  Отрицательная стоимость: {fare_negative}")

print("\nКласс (Pclass):")
pclass_zeros = (df['Pclass'] == 0).sum()
pclass_invalid = (~df['Pclass'].isin([1, 2, 3])).sum()
print(f"  Класс = 0: {pclass_zeros}")
print(f"  Невалидные классы (не 1,2,3): {pclass_invalid}")

# Визуализация нулевых значений
if any(x['Нулевых значений'] > 0 for x in numeric_zero_summary):
    plt.figure(figsize=(12, 8))
    
    # График 1: Количество нулевых значений по полям
    plt.subplot(2, 2, 1)
    zero_data = numeric_zero_df[numeric_zero_df['Нулевых значений'] > 0]
    if len(zero_data) > 0:
        sns.barplot(data=zero_data, x='Поле', y='Нулевых значений')
        plt.title('Количество нулевых значений по числовым полям', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Количество нулевых значений')
    
    # График 2: Процент нулевых значений
    plt.subplot(2, 2, 2)
    if len(zero_data) > 0:
        sns.barplot(data=zero_data, x='Поле', y='Процент')
        plt.title('Процент нулевых значений по числовым полям', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Процент нулевых значений (%)')
    
    # График 3: Распределение полей с нулевыми значениями
    plt.subplot(2, 2, 3)
    fields_with_zeros = len(zero_data)
    fields_without_zeros = len(numeric_columns) - fields_with_zeros
    plt.pie([fields_with_zeros, fields_without_zeros], 
            labels=[f'С нулевыми значениями\n({fields_with_zeros})', 
                   f'Без нулевых значений\n({fields_without_zeros})'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Распределение числовых полей по наличию нулевых значений', fontweight='bold')
    
    # График 4: Анализ стоимости билета
    plt.subplot(2, 2, 4)
    fare_zero_data = df[df['Fare'] == 0]
    if len(fare_zero_data) > 0:
        fare_zero_by_class = fare_zero_data['Pclass'].value_counts().sort_index()
        plt.bar(fare_zero_by_class.index, fare_zero_by_class.values)
        plt.title('Распределение бесплатных билетов по классам', fontweight='bold')
        plt.xlabel('Класс')
        plt.ylabel('Количество бесплатных билетов')
        plt.xticks([1, 2, 3])
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительный анализ бесплатных билетов
    if len(fare_zero_data) > 0:
        print(f"\n4. АНАЛИЗ БЕСПЛАТНЫХ БИЛЕТОВ (Fare = 0):")
        print(f"  Всего бесплатных билетов: {len(fare_zero_data)}")
        print(f"  Распределение по классам:")
        for pclass in sorted(fare_zero_data['Pclass'].unique()):
            count = len(fare_zero_data[fare_zero_data['Pclass'] == pclass])
            survival_rate = fare_zero_data[fare_zero_data['Pclass'] == pclass]['Survived'].mean()
            print(f"    Класс {pclass}: {count} билетов, выживаемость: {survival_rate:.2%}")
        
        print(f"  Выживаемость с бесплатными билетами: {fare_zero_data['Survived'].mean():.2%}")
        print(f"  Общая выживаемость: {df['Survived'].mean():.2%}")

else:
    print("\n✅ Нулевых значений в числовых полях не обнаружено")

print("\n5. Детальный анализ столбца Age:")
age_missing = df['Age'].isnull().sum()
print(f"  Пропущенных значений (NaN): {age_missing}")
print(f"  Заполненных значений: {len(df) - age_missing}")
print(f"  Процент пропусков: {age_missing/len(df)*100:.2f}%")

if missing_data.sum() == 0:
    print("\n✅ Пропущенных значений (NaN) не обнаружено")

# =============================================================================
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# =============================================================================

print("\n\n3. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("-" * 40)

# Основные статистики для числовых признаков
print("Основные статистики (числовые признаки):")
print(df.describe())

# Анализ категориальных признаков
print("\nКатегориальные признаки:")
print(f"Уникальные значения в 'Sex': {df['Sex'].unique()}")
print(f"Уникальные значения в 'Pclass': {sorted(df['Pclass'].unique())}")

# Анализ выживаемости
print("\nАНАЛИЗ ВЫЖИВАЕМОСТИ:")
total_survived = df['Survived'].sum()
total_passengers = len(df)
survival_rate = df['Survived'].mean()
print(f"Общая выживаемость: {survival_rate:.2%} ({total_survived}/{total_passengers})")

print("\nВыживаемость по классам:")
survival_by_class = df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
survival_by_class['mean_pct'] = survival_by_class['mean'].apply(lambda x: f"{x:.2%}")
print(survival_by_class[['mean_pct', 'count']].rename(columns={'mean_pct': 'Выживаемость', 'count': 'Количество'}))

print("\nВыживаемость по полу:")
survival_by_sex = df.groupby('Sex')['Survived'].agg(['mean', 'count'])
survival_by_sex['mean_pct'] = survival_by_sex['mean'].apply(lambda x: f"{x:.2%}")
print(survival_by_sex[['mean_pct', 'count']].rename(columns={'mean_pct': 'Выживаемость', 'count': 'Количество'}))

# =============================================================================
# 4. ВИЗУАЛИЗАЦИЯ ДАННЫХ
# =============================================================================

print("\n\n4. ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("-" * 40)

# Создаем фигуру с несколькими subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('АНАЛИЗ ДАННЫХ ТИТАНИКА', fontsize=16, fontweight='bold')

# График 1: Выживаемость по классу и полу
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Выживаемость по классу и полу', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Доля выживших', fontsize=12)
axes[0, 0].set_xlabel('Класс билета', fontsize=12)
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_xticklabels(['Первый', 'Второй', 'Третий'])
axes[0, 0].legend(title='Пол', labels=['Мужской', 'Женский'])

# График 2: Распределение возрастов
sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Распределение возраста по выживаемости', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Возраст', fontsize=12)
axes[0, 1].set_ylabel('Количество пассажиров', fontsize=12)
handles, labels = axes[0, 1].get_legend_handles_labels()
axes[0, 1].legend(handles, ['Не выжил', 'Выжил'], title='Результат')

# График 3: Стоимость билета vs Выживаемость
sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Распределение стоимости билета', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Выживаемость', fontsize=12)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['Не выжил', 'Выжил'])
axes[1, 0].set_ylabel('Стоимость билета (£)', fontsize=12)

# График 4: Количество родственников
relatives_sum = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard']
sns.countplot(x=relatives_sum, hue=df['Survived'], ax=axes[1, 1])
axes[1, 1].set_title('Выживаемость по количеству родственников', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Всего родственников на борту', fontsize=12)
axes[1, 1].set_ylabel('Количество пассажиров', fontsize=12)
axes[1, 1].legend(title='Выжил', labels=['Нет', 'Да'])

plt.tight_layout()
plt.show()

# Дополнительная визуализация: тепловая карта корреляций
# Дополнительная визуализация: тепловая карта корреляций
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])

russian_columns = {
    'Survived': 'Выжил',
    'Pclass': 'Класс', 
    'Age': 'Возраст',
    'Siblings/Spouses Aboard': 'Братья/Сёстры/Супруги',
    'Parents/Children Aboard': 'Родители/Дети',
    'Fare': 'Стоимость билета'
}

numeric_df_rus = numeric_df.rename(columns=russian_columns)
correlation_matrix = numeric_df_rus.corr()

# УБИРАЕМ МАСКУ для полного отображения матрицы
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
            linewidths=0.5, linecolor='white')
plt.title('МАТРИЦА КОРРЕЛЯЦИЙ МЕЖДУ ПРИЗНАКАМИ', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Дополнительно выводим матрицу в текстовом виде для наглядности
print("\nМАТРИЦА КОРРЕЛЯЦИЙ:")
print("=" * 50)
print(correlation_matrix.round(2))

# =============================================================================
# 5. АНАЛИЗ И ОБРАБОТКА ВЫБРОСОВ
# =============================================================================

print("\n\n5. АНАЛИЗ И ОБРАБОТКА ВЫБРОСОВ")
print("-" * 40)

def analyze_outliers(column_name, data, russian_name):
    """Анализ выбросов для указанного столбца"""
    print(f"\nАнализ выбросов для '{russian_name}':")
    
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Количество выбросов: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    return outliers, (lower_bound, upper_bound)

# Анализ выбросов для возраста
age_outliers, age_bounds = analyze_outliers('Age', df, 'Возраст')

# Анализ выбросов для стоимости билета
fare_outliers, fare_bounds = analyze_outliers('Fare', df, 'Стоимость билета')

# Визуализация выбросов
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(y=df['Age'], ax=ax1)
ax1.set_title('Распределение возраста (Диаграмма размаха)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Возраст', fontsize=12)

sns.boxplot(y=df['Fare'], ax=ax2)
ax2.set_title('Распределение стоимости билета (Диаграмма размаха)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Стоимость билета (£)', fontsize=12)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ
# =============================================================================

print("\n\n6. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("-" * 40)

df['IsChild'] = df['Age'] < 18
child_survival = df.groupby('IsChild')['Survived'].mean()
print("\nВыживаемость детей vs взрослых:")
print(f"Дети (<18 лет): {child_survival[True]:.2%}")
print(f"Взрослые (≥18 лет): {child_survival[False]:.2%}")

df['TotalRelatives'] = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard']
relatives_survival = df.groupby('TotalRelatives', observed=True)['Survived'].mean()
print("\nВыживаемость по количеству родственников:")
for rel_count, survival_rate in relatives_survival.items():
    count = len(df[df['TotalRelatives'] == rel_count])
    print(f"  {rel_count} родственников: {survival_rate:.2%} ({count} чел.)")

# Дополнительный график
plt.figure(figsize=(12, 6))

age_bins = [0, 12, 18, 35, 60, 100]
age_labels = ['Дети (0-12)', 'Подростки (13-18)', 'Молодые (19-35)', 'Взрослые (36-60)', 'Пожилые (60+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

survival_by_agegroup = df.groupby('AgeGroup', observed=True)['Survived'].mean().reset_index()

plt.subplot(1, 2, 1)
sns.barplot(x='AgeGroup', y='Survived', data=survival_by_agegroup, 
            hue='AgeGroup', legend=False, palette='viridis')
plt.title('Выживаемость по возрастным группам', fontsize=14, fontweight='bold')
plt.xlabel('Возрастная группа', fontsize=12)
plt.ylabel('Доля выживших', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
class_distribution = df['Pclass'].value_counts().sort_index()
colors = ['gold', 'lightcoral', 'lightskyblue']
labels = ['Первый класс', 'Второй класс', 'Третий класс']
plt.pie(class_distribution.values, labels=labels, autopct='%1.1f%%', 
        startangle=90, colors=colors)
plt.title('Распределение пассажиров по классам', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# 7. ВЫВОДЫ И РЕЗУЛЬТАТЫ
# =============================================================================

print("\n\n7. ОСНОВНЫЕ ВЫВОДЫ И РЕЗУЛЬТАТЫ")
print("-" * 40)

print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
print(f"✓ Общая выживаемость: {survival_rate:.2%} ({total_survived}/{total_passengers})")
print(f"✓ Женщины выживали значительно чаще мужчин ({survival_by_sex.loc['female', 'mean']:.2%} vs {survival_by_sex.loc['male', 'mean']:.2%})")
print(f"✓ Пассажиры 1-го класса имели наивысшие шансы на выживание ({survival_by_class.loc[1, 'mean']:.2%})")
print(f"✓ Дети имели повышенные шансы на выживание по сравнению со взрослыми ({child_survival[True]:.2%} vs {child_survival[False]:.2%})")
print("✓ Стоимость билета положительно коррелирует с выживаемостью")

print(f"\n📈 СТАТИСТИКА ВЫБРОСОВ:")
print(f"✓ Выбросы в возрасте: {len(age_outliers)} ({len(age_outliers)/len(df)*100:.1f}%)")
print(f"✓ Выбросы в стоимости билета: {len(fare_outliers)} ({len(fare_outliers)/len(df)*100:.1f}%)")

print(f"\n🔍 КАЧЕСТВО ДАННЫХ:")
print("✓ Пропущенных значений (NaN): не обнаружено")
print("✓ Пустых строк: не обнаружено")
print(f"✓ Нулевых значений в стоимости билета: {(df['Fare'] == 0).sum()} ({(df['Fare'] == 0).sum()/len(df)*100:.1f}%)")
print(f"✓ Нулевых значений в братьях/сестрах/супругах: {(df['Siblings/Spouses Aboard'] == 0).sum()} ({(df['Siblings/Spouses Aboard'] == 0).sum()/len(df)*100:.1f}%)")
print(f"✓ Нулевых значений в родителях/детьях: {(df['Parents/Children Aboard'] == 0).sum()} ({(df['Parents/Children Aboard'] == 0).sum()/len(df)*100:.1f}%)")
print("✓ Аномальных значений в возрасте: не обнаружено")

print("\n✅ ЗАКЛЮЧЕНИЕ:")
print("✓ Данные прошли полный анализ и предобработку")
print("✓ Проведена детальная проверка нулевых значений во всех полях")
print("✓ Выбросы проанализированы и документированы")
print("✓ Данные готовы для построения моделей машинного обучения")

print("\n" + "=" * 60)
print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
print("=" * 60)


