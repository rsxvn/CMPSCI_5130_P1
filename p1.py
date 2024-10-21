import pandas as pd
from pulp import *


def main():
    # Read the CSV file into a pandas DataFrame
    file_name = 'tnn_data_1200_clicks.csv'
    click_val = int(file_name.split('_')[2])

    data = pd.read_csv(file_name)

    # Converting DF columns to list
    articles = data['Article'].tolist()
    reporters = data['Reporter'].astype(str).unique().tolist()
    types = data['Type'].unique().tolist()

    # Dictionary to map everything to articles
    cost = dict(zip(data['Article'], data['Cost']))
    clicks = dict(zip(data['Article'], data['Clicks']))
    article_type = dict(zip(data['Article'], data['Type']))
    article_reporter = dict(zip(data['Article'], data['Reporter'].astype(str)))

    print(articles, reporters)

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

    repeated_type_vars = LpVariable.dict('dv_extra_article',
                                         types,
                                         cat=LpBinary)

    article_reporter_dic = {}
    for article in articles:
        article_reporter_dic[article] = {}
        for reporter in reporters:
            if article_reporter[article] == reporter:
                article_reporter_dic[article][reporter] = 1
            else:
                article_reporter_dic[article][reporter] = 0

    article_type_dic = {}
    for article in articles:
        article_type_dic[article] = {}
        for ty in types:
            if article_type[article] == ty:
                article_type_dic[article][ty] = 1
            else:
                article_type_dic[article][ty] = 0

    # Constraint 1
    for reporter in reporters:
        lp += lpSum(article_reporter_dic[a][reporter] * selected_report_vars[a]
                    for a in articles) >= 1
    # Constraint 2
    lp += lpSum(clicks[a] * selected_report_vars[a]
                for a in articles) >= click_val

    # Constraint 3
    for typ in types:
        lp += lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                    for a in articles) >= 1

    for typ in types:
        lp += (lpSum(selected_report_vars[a] for a in articles)
               - 2 * lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                           for a in articles)) >= 0

    # Constraint 5
    for reporter in reporters:
        lp += (lpSum(article_reporter_dic[a][reporter]
                     * selected_report_vars[a]
                     for a in articles)
               - 1 == extra_article_vars[reporter])

    # Constraint 6

    num_of_articles = len(articles)
    for typ in types:
        # Constraint 6.1
        lp += (lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                     for a in articles)
               - (num_of_articles - 1) * repeated_type_vars[typ] <= 1)

        # Constraint 6.2
        lp += (lpSum(article_type_dic[a][typ] * selected_report_vars[a]
                     for a in articles)
               - 2 * repeated_type_vars[typ] >= 0)

    lp += ((lpSum(selected_report_vars[a] * cost[a] for a in articles))
           + lpSum(
                100 * extra_article_vars[reporter] for reporter in reporters)
           - lpSum(115 * repeated_type_vars[typ] for typ in types))

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

    #print(repeated_type_vars['L'].varValue, selected_report_vars['A1'].varValue)
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
