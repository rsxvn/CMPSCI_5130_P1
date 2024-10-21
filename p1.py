from typing import Any, Dict

import pandas as pd
from pandas import DataFrame
from pulp import *


def main():
    # Read the CSV file into a pandas DataFrame
    file_name: str = 'tnn_data_4200_clicks.csv'
    click_val: int = int(file_name.split('_')[2])

    data: DataFrame = pd.read_csv(file_name)

    # Converting DF columns to list
    articles: list[str] = data['Article'].tolist()
    reporters: list[str] = data['Reporter'].astype(str).unique().tolist()
    types: list[str] = data['Type'].unique().tolist()

    # Dictionary to map everything to articles
    cost: dict[str, int] = dict(zip(data['Article'], data['Cost']))
    clicks: dict[str, int] = dict(zip(data['Article'], data['Clicks']))
    article_type: dict[str, str] = dict(zip(data['Article'], data['Type']))
    article_reporter: dict[str, str] = dict(zip(data['Article'],
                                                data['Reporter'].astype(str)))

    # Initialize the problem
    lp = LpProblem("Project_1", LpMinimize)

    # Decision Variables
    selected_report_vars = LpVariable.dict('dv_selected_article',
                                           articles,
                                           cat=LpBinary)

    extra_article_vars = LpVariable.dict('dv_extra_article',
                                         reporters,
                                         lowBound=0,
                                         cat=LpInteger)

    repeated_type_vars = LpVariable.dict('dv_repeated_type',
                                         types,
                                         cat=LpBinary)

    # Creating a dictionary to show similar to the mathematical formula
    article_reporter_dic: dict[str, dict[str, int]] = {}
    for article in articles:
        article_reporter_dic[article] = {}
        for reporter in reporters:
            if article_reporter[article] == reporter:
                article_reporter_dic[article][reporter] = 1
            else:
                article_reporter_dic[article][reporter] = 0

    # Creating a dictionary to show similar to the mathematical formula
    article_type_dic: dict[str, dict[str, int]] = {}
    for article in articles:
        article_type_dic[article] = {}
        for ty in types:
            if article_type[article] == ty:
                article_type_dic[article][ty] = 1
            else:
                article_type_dic[article][ty] = 0

    # Constraint 1
    for reporter in reporters:
        lp += (lpSum(article_reporter_dic[a][reporter] * selected_report_vars[a]
                    for a in articles) >= 1, f"C1[{reporter}]")
    # Constraint 2
    lp += (lpSum(clicks[a] * selected_report_vars[a]
                for a in articles) >= click_val, f"C2")

    # Constraint 3
    for typ in types:
        lp += (lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                    for a in articles) >= 1, f"C3[{typ}]")

    # Constraint 4
    for typ in types:
        lp += ((lpSum(selected_report_vars[a] for a in articles)
               - 2 * lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                           for a in articles)) >= 0, f"C4[{typ}]")

    # Constraint 5
    for reporter in reporters:
        lp += (lpSum(article_reporter_dic[a][reporter]
                     * selected_report_vars[a]
                     for a in articles)
               - 1 == extra_article_vars[reporter], f"C5[{reporter}]")

    # Constraint 6
    # Broken the constraint in two parts to ensure 1 or 0
    num_of_articles = len(articles)
    for typ in types:
        # Constraint 6.1
        lp += (lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                     for a in articles)
               - (num_of_articles - 1) * repeated_type_vars[typ] <= 1, f"C6.1[{typ}]")

        # Constraint 6.2
        lp += (lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                     for a in articles)
               - 2 * repeated_type_vars[typ] >= 0, f"C6.2[{typ}]")

    # Objective function with constraint 5 & 6, uncomment to use
    lp += ((lpSum(selected_report_vars[a] * cost[a] for a in articles))
           + lpSum(
                100 * extra_article_vars[reporter] for reporter in reporters)
           - lpSum(115 * repeated_type_vars[typ] for typ in types))

    # Objective function without constraint 5 & 6, uncomment to use
    #lp += (lpSum(selected_report_vars[a] * cost[a] for a in articles))

    # Solve the problem
    lp.solve()

    # Print the status
    print("Status:", LpStatus[lp.status])

    # Print the objective function and constraints
    print("\nObjective Function:")
    print(lp.objective)

    print("\nConstraints:")
    for name, constraint in lp.constraints.items():
        print(f"{name}: {constraint}")

    selected_articles = [i for i in articles
                         if selected_report_vars[i].varValue == 1]

    print("\nSelected articles:")
    for i in selected_articles:
        print(
            f"Article {i}: "
            f"Reporter {article_reporter[i]}"
            f", Type {article_type[i]}"
            f", Cost {cost[i]}"
            f", Clicks {clicks[i]}")

    # Print the total cost
    print("\nTotal Cost:", value(lp.objective))


if __name__ == "__main__":
    main()
