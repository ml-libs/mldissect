from terminaltables import AsciiTable


class Explanation:

    def __init__(self, data):
        self.data = data
        self.digits = 4

    def _make_table_data(self,):
        table_data = []
        header = ('Feature', 'Value', 'Contribution')
        table_data.append(header)

        baseline = ('baseline', "-", str(self.data.intercept[0][0]))
        table_data.append(baseline)
        d = self.data
        for i in range(len(self.data.columns)):
            row = (
                d.columns[i],
                d.values[i],
                round(d.contribution[0][i], self.digits),
            )
            table_data.append(row)
        return table_data

    def print(self):
        table_data = self._make_table_data()
        table = AsciiTable(table_data)
        print(table.table)

    def raw_result(self):
        return self.data
