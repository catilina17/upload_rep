import pandas as pd
import os
from utils import excel_utils as ex
from mappings.mapping_products import products_map
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from calculateur.calc_params import model_params as mod
import traceback
import logging
logger = logging.getLogger(__name__)

class Models_Reports():
    """
    Formate les données
    """
    def __init__(self, sim_params, custom_report, benchmark, is_tci):
        self.xlDatabase = 1
        self.xlRowField = 1
        self.xlSum = -4157
        self.xlDown = -4121
        self.xlFormulas = -4123
        self.xlCenter = -4108
        self.custom_report = custom_report
        self.benchmark_file_path = benchmark
        self.is_tci = is_tci
        self.indics = ["LEM", "TEM", "LMN"]
        self.TEMPLATE_REPORT_NAME = self.get_model_report_file_name(sim_params)

    def get_model_report_file_name(self, sim_params):
        dar_stamp = str(sim_params.dar.year) + '{:02d}'.format(sim_params.dar.month) + str(sim_params.dar.day)
        model_tag = os.path.splitext(sim_params.model)[0]
        return "MODELS_SYNTHESIS_%s_%s_%s%s.xlsb" % (dar_stamp, model_tag, sim_params.rate_scenario, "_TCI" if self.is_tci else "")
    def set_cls_data(self, cls_data):
        self.cls_data = cls_data

    def set_cls_mod_params(self, cls_mod_params):
        self.cls_mod_params = cls_mod_params

    def create_new_wb(self):
        path_output_file = os.path.join(self.custom_report.output_folder, self.TEMPLATE_REPORT_NAME)
        try:
            ex.xl.Workbooks(self.TEMPLATE_REPORT_NAME).Close(False)
        except:
            pass
        if not os.path.exists(path_output_file):
            report_wb = ex.xl.Workbooks.Add()
            report_wb.SaveAs(path_output_file, 50)
        else:
            report_wb = ex.try_close_open(path_output_file)

        return report_wb

    def generate_models_synthesis_report(self, product_name, etab):
        report_wb = None
        name_ws = products_map[product_name]
        try:
            report_wb = self.create_new_wb()
            try:
                report_wb.Sheets(name_ws)
            except:
                if report_wb.Sheets.Count == 1:
                    report_wb.Sheets(report_wb.Sheets.Count).Name = name_ws
                else:
                    ws = report_wb.Sheets.Add(None, report_wb.Worksheets(report_wb.Sheets.Count))
                    ws.Name = name_ws

            if os.path.exists(self.benchmark_file_path):
                bench_wb = ex.try_close_open(self.benchmark_file_path, read_only=True)
                benchmark= True
                try:
                    name = products_map[product_name]
                    bench_wb_sheet_model = bench_wb.Sheets(name)
                except:
                    logger.info("  No model %s exists in the benchmark file" % products_map[product_name])
                    benchmark = False
            else:
                benchmark = False

            report_wb.Sheets.Add(None, report_wb.Worksheets(name_ws))
            data_sheet = report_wb.Worksheets(name_ws).Next
            data_sheet.Name = "DATA_%s_%s" % (product_name.upper(), etab)

            data_report = self.custom_report.data_report[self.custom_report.data_report[pa.NC_PA_IND03].isin(self.indics)].copy()

            if len(data_report) > 0:
                ex.write_df_to_address_chunk(data_report, data_sheet, (1, 1), header=True, chunk=10000)
                leap = 35
                sheet_model = report_wb.Sheets(name_ws)
                dest_range = self.get_dest_range(sheet_model, leap=leap)
                dest_range.Value = etab
                dest_range.Font.Bold = True
                dest_range.Font.Size = 13
                dest_range.HorizontalAlignment = self.xlCenter
                dest_range.VerticalAlignment = self.xlCenter

                pc = report_wb.PivotCaches().Create(SourceType=self.xlDatabase, SourceData="%s!R1C1:R1048576C%s" % (data_sheet.Name,data_report.shape[1]))

                pivot_range = dest_range.Offset(6)
                pivot_table = pc.CreatePivotTable(
                    TableDestination="'%s'!R%sC%s" % (sheet_model.Name,pivot_range.Row, pivot_range.Column),
                    TableName="TCD %s %s" % (product_name.upper(), etab))

                pivot_table.PivotFields("VERSION").Orientation = self.xlRowField
                pivot_table.PivotFields("VERSION").Position = 1

                pivot_table.PivotFields(pa.NC_PA_IND03).Orientation = 1
                pivot_table.PivotFields(pa.NC_PA_IND03).Position = self.xlRowField

                num_cols = ["A1", "A2", "A3", "A4", "A5", "M0", "M1", "M10", "M20", "M60", "M120"]
                for num_col in num_cols:
                    pivot_table.AddDataField(pivot_table.PivotFields(num_col), num_col + " ", self.xlSum)

                pivot_table.ColumnGrand = False
                pivot_table.RowGrand = False

                pivot_table.PivotFields("VERSION").Subtotals = [False] * (len(num_cols) + 1)
                pivot_table.PivotFields("IND03").Subtotals = [False] * (len(num_cols) + 1)

                sheet_model.Range("A%s:CW%s" % (pivot_range.Row, pivot_range.Row + 11)).Style = "Comma"

                lem_row = (pivot_range.Row + 14)
                mni_row = (pivot_range.Row + 15)
                tem_row = (pivot_range.Row + 16)
                lem_diff_row = (pivot_range.Row + 18)
                mni_diff_row = (pivot_range.Row + 19)
                tem_diff_row = (pivot_range.Row + 20)
                lem_bench = (pivot_range.Row + 22)
                mni_bench = (pivot_range.Row + 23)
                tem_bench = (pivot_range.Row + 24)

                if benchmark:
                    if (etab == str(bench_wb_sheet_model.Range("A1").Value) and dest_range.Row == leap):
                        correc = -leap
                    elif (etab == str(bench_wb_sheet_model.Range("A35").Value) and dest_range.Row == 1):
                        correc = leap
                    else:
                        correc = 0

                for lem_r, mni_r, tem_r, formula, format, is_benchmark in\
                        zip([lem_row, lem_diff_row, lem_bench], [mni_row, mni_diff_row, mni_bench],
                            [tem_row, tem_diff_row, tem_bench], ["=B%s/B%s-1","=B%s-B%s",""],
                            ["Percent", "Comma", "Percent"], [False, False, True]):

                    if not is_benchmark:
                        sheet_model.Range("A%s" % lem_r).Value = "DIFF%sLEM" %(" % " if format=="Percent" else " ")
                        sheet_model.Range("A%s" % mni_r).Value = "DIFF%sLMN" %(" % "  if format=="Percent" else " ")
                        sheet_model.Range("A%s" % tem_r).Value = "DIFF%sTEM" %(" % "  if format=="Percent" else " ")
                    else:
                        sheet_model.Range("A%s" % lem_r).Value = "EVOL LEM vs. BENCH"
                        sheet_model.Range("A%s" % mni_r).Value = "EVOL LMN vs. BENCH"
                        sheet_model.Range("A%s" % tem_r).Value = "EVOL TEM vs. BENCH"

                    sheet_model.Range("A%s:A%s" %(lem_r, tem_r)).Font.Bold = True
                    sheet_model.Range("A%s:A%s" %(lem_r, tem_r)).HorizontalAlignment = self.xlCenter
                    sheet_model.Range("A%s:A%s" %(lem_r, tem_r)).VerticalAlignment = self.xlCenter

                    if not is_benchmark:
                        sheet_model.Range("B%s" % lem_r).Formula = formula % (pivot_range.Row + 4, pivot_range.Row + 3)
                        sheet_model.Range("B%s" % mni_r).Formula = formula % (pivot_range.Row + 7, pivot_range.Row + 6)
                        sheet_model.Range("B%s" % tem_r).Formula = formula % (pivot_range.Row + 10, pivot_range.Row + 9)
                        analysis_table_range = sheet_model.Range("B%s:L%s" % (lem_r, tem_r))
                        sheet_model.Range("B%s:B%s" % (lem_r, tem_r)).Copy()
                        analysis_table_range.PasteSpecial(self.xlFormulas)
                    elif benchmark:
                        sheet_model.Calculate()
                        r1 = bench_wb_sheet_model.Range("B%s" % (lem_row + correc))
                        r2 = bench_wb_sheet_model.Range("L%s" % (tem_row + correc))
                        vals_benchmark = bench_wb_sheet_model.Range(r1, r2).Value
                        vals_benchmark = ex.RangeToDataframe(vals_benchmark, header=False)
                        r1 = sheet_model.Range("B%s" % (lem_row))
                        r2 = sheet_model.Range("L%s" % (tem_row))
                        vals = sheet_model.Range(r1, r2).Value
                        vals = ex.RangeToDataframe(vals, header=False)
                        evol = pd.DataFrame(vals.values - vals_benchmark.values)
                        ex.write_df_to_address_chunk(evol, sheet_model, (lem_r, 2), header=False)
                        analysis_table_range = sheet_model.Range("B%s:L%s" % (lem_r, tem_r))
                    else:
                        analysis_table_range = sheet_model.Range("B%s:L%s" % (lem_r, tem_r))

                    if format =="Comma":
                        analysis_table_range.Style = 'Comma'
                    else:
                        analysis_table_range.NumberFormat = "0,0000%"

                for address, type in zip(["B%s:L%s" %(lem_row, lem_row), "B%s:L%s" %(mni_row, mni_row), "B%s:L%s" %(tem_row, tem_row),
                                          "B%s:L%s" %(lem_diff_row, lem_diff_row), "B%s:L%s" %(mni_diff_row, mni_diff_row),"B%s:L%s" %(tem_diff_row, tem_diff_row),
                                          "B%s:L%s" %(lem_bench, lem_bench), "B%s:L%s" %(mni_bench, mni_bench),"B%s:L%s" %(tem_bench, tem_bench)],
                                          [1, 1, 1, 0, 0, 0, 2, 2, 2]):
                    diff_range = sheet_model.Range(address)
                    diff_range.HorizontalAlignment = self.xlCenter
                    diff_range.VerticalAlignment = self.xlCenter

                    if type in [1, 2]:
                        threshold = "=0,005" if type ==1 else "=0,00001"
                        type_comp = 7 if type==1 else 5
                        diff_range.FormatConditions.Add(1, type_comp, threshold)
                        diff_range.FormatConditions(diff_range.FormatConditions.Count).SetFirstPriority()
                        diff_range.FormatConditions(1).Font.Bold = True
                        diff_range.FormatConditions(1).Font.Italic = False
                        diff_range.FormatConditions(1).Font.ThemeColor = 1
                        diff_range.FormatConditions(1).Font.TintAndShade = 0

                        diff_range.FormatConditions(1).Interior.PatternColorIndex = -4105
                        diff_range.FormatConditions(1).Interior.Color = 8224247
                        diff_range.FormatConditions(1).Interior.TintAndShade = 0
                        diff_range.FormatConditions(1).StopIfTrue = False

                        threshold = "=-0,005" if type ==1 else "=-0,00001"
                        type_comp = 8 if type==1 else 6
                        diff_range.FormatConditions.Add(1, type_comp, threshold)
                        diff_range.FormatConditions(diff_range.FormatConditions.Count).SetFirstPriority()
                        diff_range.FormatConditions(1).Font.Bold = True
                        diff_range.FormatConditions(1).Font.Italic = False
                        diff_range.FormatConditions(1).Font.ThemeColor = 1
                        diff_range.FormatConditions(1).Font.TintAndShade = 0

                        color = 8224247 if type==1 else 9559572
                        diff_range.FormatConditions(1).Interior.PatternColorIndex = -4105
                        diff_range.FormatConditions(1).Interior.Color = color
                        diff_range.FormatConditions(1).Interior.TintAndShade = 0
                        diff_range.FormatConditions(1).StopIfTrue = False

                sheet_model.Calculate()

                self.remove_gridlines(sheet_model)
                data_sheet.Visible = False
                report_wb.Close(True)
                if benchmark:
                    try:
                        bench_wb.Close(False)
                    except:
                        pass

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            try:
                report_wb.Close(False)
            except:
                pass
            try:
                bench_wb.Close(False)
            except:
                pass

    def get_dest_range(self, sheet_model, leap=30):
        dest_range = sheet_model.Range("A1")
        while(dest_range.Value is not None):
            dest_range = dest_range.offset(leap)

        return dest_range

    def remove_gridlines(self, sheet):
        for view in sheet.Parent.Windows(1).SheetViews:
            if view.Sheet.Name == sheet.Name:
                view.DisplayGridlines = False
                return
        return
