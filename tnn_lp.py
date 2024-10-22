__author__: str = "Rumandeep Singh"
__email__: str = "rsxvn@umsl.edu"
__instructor__: str = "Dr. Azim Ahmadzadeh"
__date__: str = "10/21/2024"

import pandas as pd
from pulp import *


def main():
    """
    Yet to update  the details.
    """
    # Read the CSV file into a pandas DataFrame
    file_name: str = 'tnn_data_4200_clicks.csv'
    click_val: int = int(file_name.split('_')[2])

    data: pd.DataFrame = pd.read_csv(file_name)

    # Converting DF columns to list
    articles: list[str] = data['Article'].tolist()
    reporters: list[str] = data['Reporter'].astype(str).unique().tolist()
    types: list[str] = data['Type'].unique().tolist()

    # Dictionary to map everything to articles
    cost: dict[str, int] = dict(zip(data['Article'],
                                    data['Cost']))

    clicks: dict[str, int] = dict(zip(data['Article'],
                                      data['Clicks']))

    article_type: dict[str, str] = dict(zip(data['Article'],
                                            data['Type']))

    article_reporter: dict[str, str] = dict(zip(data['Article'],
                                                data['Reporter'].astype(str)))

    # Initialize the problem
    lp = LpProblem("Project_1", LpMinimize)

    # Decision Variables

    # Binary variable for selection of article:
    selected_report_vars = LpVariable.dict('dv_selected_article',
                                           articles,
                                           cat=LpBinary)

    # Integer variable for keeping track of extra articles for each reporter
    extra_article_vars = LpVariable.dict('dv_extra_article',
                                         reporters,
                                         lowBound=0,
                                         cat=LpInteger)

    # Binary variable to check if more than 1 articles for all types
    repeated_type_vars = LpVariable.dict('dv_repeated_type',
                                         types,
                                         cat=LpBinary)

    # Auxilary Variables

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
        for typ in types:
            if article_type[article] == typ:
                article_type_dic[article][typ] = 1
            else:
                article_type_dic[article][typ] = 0

    # Constraints

    # Constraint 1
    for reporter in reporters:
        lp += (lpSum(article_reporter_dic[article][reporter]
                     * selected_report_vars[article]
                     for article in articles)
               >= 1, f"C1[{reporter}]")

    # Constraint 2
    lp += (lpSum(clicks[article] * selected_report_vars[article]
                 for article in articles)
           >= click_val, f"C2")

    # Constraint 3
    for typ in types:
        lp += (lpSum(article_type_dic[article][typ]
                     * selected_report_vars[article]
                     for article in articles)
               >= 1, f"C3[{typ}]")

    # Constraint 4
    for typ in types:
        lp += ((lpSum(selected_report_vars[article] for article in articles)
                - 2 * lpSum(article_type_dic[article][typ]
                            * selected_report_vars[article]
                            for article in articles))
               >= 0, f"C4[{typ}]")

    # Constraint 5
    for reporter in reporters:
        lp += (lpSum(article_reporter_dic[article][reporter]
                     * selected_report_vars[article]
                     for article in articles) - extra_article_vars[reporter]
               == 1, f"C5[{reporter}]")

    # Constraint 6
    # Broken the constraint in two parts to ensure 1 or 0
    num_of_articles = len(articles)
    for typ in types:
        # Constraint 6.1
        lp += (lpSum(article_type_dic[article][typ]
                     * selected_report_vars[article]
                     for article in articles)
               - (num_of_articles - 1) * repeated_type_vars[typ]
               <= 1, f"C6.1[{typ}]")

        # Constraint 6.2
        lp += (lpSum(article_type_dic[article][typ]
                     * selected_report_vars[article]
                     for article in articles)
               - 2 * repeated_type_vars[typ]
               >= 0, f"C6.2[{typ}]")

    # Objective function with constraint 5 & 6, uncomment to use
    lp += ((lpSum(selected_report_vars[article]
                  * cost[article]
                  for article in articles))
           + lpSum(
                100 * extra_article_vars[reporter]
                for reporter in reporters)
           - lpSum(115 * repeated_type_vars[typ]
                   for typ in types))

    # Objective function without constraint 5 & 6, uncomment to use
    # lp += (lpSum(selected_report_vars[article]
    #       * cost[article]
    #       for article in articles))

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

    selected_articles = [article for article in articles
                         if selected_report_vars[article].varValue == 1]

    print("\nSelected articles:")
    for article in selected_articles:
        print(
            f"Article {article}: "
            f"Reporter {article_reporter[article]}"
            f", Type {article_type[article]}"
            f", Cost {cost[article]}"
            f", Clicks {clicks[article]}")

    # Print the total cost
    print("\nTotal Cost:", value(lp.objective))


if __name__ == "__main__":
    main()
