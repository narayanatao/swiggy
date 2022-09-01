import cerberus
import json
import random
import TAPPconfig as cfg
from dateutil.parser import parse
import collections.abc
from cerberus import Validator

rules_json_filepath = cfg.getValidationRulesPath()
# rules_json_filepath = "VALIDATION_RULES.json"
def read_rules(json_file_path = rules_json_filepath):
    """
    """
    rules = {}
    with open(json_file_path) as json_file:
        rules = json.load(json_file)
    return rules


class MyValidator(Validator):
    def _validate_type_Alphanumeric(self, value):
        """
        By defalut everything is Alphanumeric
        """
        return True

    def _normalize_coerce_convertAmount(self, value):
        try:
            v = float(str(value).replace(',',''))
            return v
        except Exception as e:
            print(e)
            return value

    def _validate_type_Amount(self, value):
        """
        Test whether the value is Amount (Float/Integer) or not
        Replace comma(',') with ''
        """
        try:
            v = float(str(value).replace(',',''))
            return True
        except Exception as e:
            return False

    def _validate_type_Number(self, value):
        """
        Test whether the value is Number or not
        Replace comma(',') with ''
        """
        return str(value).isnumeric()

    def _validate_type_Date(self, value):
        """
        Test whether the value is Number or not
        Replace comma(',') with ''
        """
        return is_date(value)


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def update_rule(default_rule, vendor_specific_rule):
    """
    Method to update Default Rule based on Vendor Specific Rule
    """
    for k, v in vendor_specific_rule.items():
        if isinstance(v, collections.abc.Mapping):
            default_rule[k] = update_rule(default_rule.get(k, {}), v)
        else:
            default_rule[k] = v
    return default_rule

def convert(val):
    """
    """
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val.replace(',',''))
        except ValueError:
            pass


def format_prediction_for_validation(pred):
    """
    """
    data={}
    for key, value in pred.items():
        if value is not None:
            if key != "lineItemPrediction":
                if value is not None:
                    data[key] = value['text']
            else:
                l = []
                for page, page_pred in value.items():
                    for row, row_pred in page_pred.items():
                        dict_pred_line_item = {"page_num" : page, "row_num": row}
                        for line_items in row_pred:
                            for f,v in line_items.items():
                                dict_pred_line_item[f] = v['text']
                        l.append(dict_pred_line_item)
                data[key] = l
    return data


def validate_final_output(pred, vendor_id):
    """
    """
    print("validate_final_output called for VENDOR:", vendor_id)
    defined_rules = read_rules()
    default_rule = defined_rules["DEFAULT"]
    print("Default Rule:", default_rule)

    schema = default_rule
    if vendor_id in defined_rules:
        vendor_specific_rule = defined_rules[vendor_id]
        schema = update_rule(default_rule, vendor_specific_rule)
    print("Updated Rule:", schema)


    document = format_prediction_for_validation(pred)
    print(document)
    try:
        # norm = MyNormalizer()
        # document = norm.normalized(document, {"totalAmount" : {"coerce": "abc"}})
        v = MyValidator(schema, allow_unknown=True)
        document = v.normalized(document)
        v.validate(document)
        dict_error = v.errors
        print(dict_error)
        if dict_error:
            modified_pred = {}
            for key, val in pred.items():
                if (val is not None) and (key in dict_error):
                    if key != "lineItemPrediction" :
                        print("Decreasing Confidence for ", key, " due to ", dict_error[key])
                        new_conf = 0.40
                        val['final_confidence_score'] = new_conf
                        modified_pred[key] = val
                    else:
                        print("Decreasing Confidence for ", key, " due to ", dict_error[key])
                        items_identified = []
                        line_item_errors = dict_error[key][0]
                        for index, items in line_item_errors.items():
                            page_num = document['lineItemPrediction'][index]['page_num']
                            row_num = document['lineItemPrediction'][index]['row_num']
                            fields = list(items[0].keys())
                            for f in fields:
                                items_identified.append((page_num, row_num, f))

                        print("Decrease LineItem Confidence for:", items_identified)
                        new_lineItem_pred = {}
                        for page, page_pred in val.items():
                            new_page_pred = {}
                            for row, row_pred in page_pred.items():
                                new_row_pred = []
                                for line_items in row_pred:
                                    for f,v in line_items.items():
                                        if (page, row, f) in items_identified:
                                            new_conf = 0.40
                                            v['model_confidence'] = new_conf
                                            new_row_pred.append({f: v})
                                        else:
                                            new_row_pred.append(line_items)
                                new_page_pred[row] = new_row_pred
                            new_lineItem_pred[page] = new_page_pred
                        modified_pred["lineItemPrediction"] = new_lineItem_pred
                else:
                    modified_pred[key] = val
            return modified_pred
    except Exception as e:
        print("<><><><><><><><><><><><><><><><><>")
        print(e)

    return pred
