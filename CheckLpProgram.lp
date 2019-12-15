\* My LP Problem *\
Maximize
OBJ: 10 Itemsets_x1 + 20 Itemsets_x2 + 30 Itemsets_x3 - 4 Products_beer
 - Products_cheese - 5 Products_cola - 3 Products_peanuts
Subject To
1Constraint: Products_beer + Products_cheese + Products_cola
 + Products_peanuts <= 3
2Constraint: - Itemsets_x2 + Products_peanuts >= 0
3Constraint: - Itemsets_x2 + Products_cheese >= 0
4Constraint: - Itemsets_x3 + Products_peanuts >= 0
5Constraint: - Itemsets_x3 + Products_beer >= 0
6Constraint: - Itemsets_x1 + Products_cola >= 0
7Constraint: - Itemsets_x1 + Products_peanuts >= 0
Binaries
Itemsets_x1
Itemsets_x2
Itemsets_x3
Products_beer
Products_cheese
Products_cola
Products_peanuts
End
