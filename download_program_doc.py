import urllib.parse
import requests
import json

def generate_url(data_dict, program_number):

    if(data_dict.get("DOC_NAME") is None):
        data_dict["DOC_NAME"] = ""
    if(data_dict.get("ED_DOC_INFO") is None):
        data_dict["ED_DOC_INFO"] = ""
    if(data_dict.get("DOC_PAGE_NO") is None):
        data_dict["DOC_PAGE_NO"] = 0
        # print (data_dict)
        

    base_url = "https://mavat.iplan.gov.il/rest/api/Attacments/"
    query_params = {
        "eid": int(data_dict.get("ID")),
        "fn": f"{program_number}_{urllib.parse.quote(data_dict.get('DOC_NAME', ''))}_{urllib.parse.quote(data_dict.get('ED_DOC_INFO', ''))}_{int(data_dict.get('DOC_PAGE_NO', 0))}.pdf",
        "edn": "temp-default",
        "pn": program_number
    }

    return f"{base_url}?{'&'.join([f'{k}={v}' for k, v in query_params.items()])}"


#takes a program number and returns the xplan number
def extract_xplan_number(program_number):
    # URL from which to fetch the JSON data
    url = "https://ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer/1/query?f=json&where=pl_number%20LIKE%20%27" + program_number + "%25%27&returnGeometry=true&spatialRel=esriSpatialRelIntersects&outFields=pl_number%2Cpl_name%2Cpl_url%2Cpl_area_dunam%2Cquantity_delta_120%2Cstation_desc%2Cpl_date_advertise%2Cpl_date_8%2Cplan_county_name%2Cpl_landuse_string&orderByFields=pl_number"
    #print(url)
    # Send a GET request to the URL
    response = requests.get(url)
    # Parse the response as JSON
    parsed_data = json.loads(response.text)
    #print(parsed_data)

    # Load the data into a Python object
    # Extract the URL
    url = parsed_data["features"][0]["attributes"]["pl_url"]
    # Extract the number from the URL
    iplan_number = url.split('/')[-2]

    return iplan_number

#takes a program number and returns a list of urls for the documents of the program
def program_doc_url(iplan_number , doc_type , program_number):#change!!!!!!!!!!!!
    url2 = "https://mavat.iplan.gov.il/rest/api/SV4/1?mid=" + iplan_number
    response = requests.get(url2)
    parsed_data = json.loads(response.text)
    generated_urls = []
    # print(parsed_data.keys())
    # print("row",parsed_data[doc_type][0])
    for row in parsed_data[doc_type]:
        # for key in row.keys():
        #     if row[key] is None:
        #         del row[key]
        generated_url = generate_url(row, program_number)
        generated_urls.append(generated_url)
        # print(generated_url)
    return generated_urls

if __name__ == "__main__":

    program_number = "101-0699561"
    iplan_number = extract_xplan_number(program_number)

    doc_type = "rsPlanDocs"

    # 'rsTopic', 'rsPlanDocs', 'rsPlanDocsAdd'
    print("xplan_number : " + iplan_number)


    generated_urls = program_doc_url(iplan_number , doc_type, program_number)
    for generated_url in generated_urls:
        print(generated_url)


if __name__ == "__main__test__":

    test = False

    if test:
        # Example input
        example_dict = {
            "ID": 1000859545595.0,
            "DOC_NAME": "תדפיס הוראות התכנית",
            "ED_DOC_INFO": "חתום להפקדה",
            "DOC_PAGE_NO": 1.0
        }

        # Generate the URL
        # generated_url = generate_url(example_dict, program_number)
        # print(generated_url)


        example_dict = {"ID": 1000859545593.0, "DOC_NAME": "תדפיס תשריט מצב מוצע",
                        "ED_DOC_INFO": "תשריט מצב מוצע - חתום להפקדה", "INTERNAL_OPEN_DATE": "05/02/2023",
                        "EDITING_DATE": "05/02/2023", "DOC_PAGES": 1.0, "DOC_PAGE_NO": 1.0,
                        "ATTACHMENT_ID": 1000022998509.0, "FILE_TYPE": "pdf       ", "RUB_CODE": 250.0,
                        "RUB_DESC": "מסמכים חתומים", "ORD": 250.0, "ED_ENTITY_TYPE": 310.0, "ED_DOC_SOURCE": 11.0,
                        "ED_DOC_TYPE": 40002.0, "FILE_DATA": {"edId": None,
                                                              "edNum": "467E52ED993DEA53D24132658232B16B8AA526DFBFF265ACECBE3E083CF80404",
                                                              "fname": "DOC_1000859545593.pdf", "ficon": "ft/file_PDF.gif",
                                                              "attExist": True},
                        "PLAN_ENTITY_DOC_NUM": "AC245EC44A1569A6BB1925542B88CECD7C4D787E9F6F40B558F00B1CEF85EFCE"}
        example_dict = {"ID": 1000859545567.0, "DOC_NAME": "תנועה",
                        "ED_DOC_INFO": "נספח מס' 03.1 - נספח תנועה שלב א' - חתום להפקדה",
                        "INTERNAL_OPEN_DATE": "05/02/2023", "EDITING_DATE": "05/02/2023", "DOC_PAGES": 0.0,
                        "DOC_PAGE_NO": 0.0, "ATTACHMENT_ID": 1000022998463.0, "FILE_TYPE": "pdf       ", "RUB_CODE": 250.0,
                        "RUB_DESC": "מסמכים חתומים", "ORD": 250.0, "ED_ENTITY_TYPE": 310.0, "ED_DOC_SOURCE": 11.0,
                        "ED_DOC_TYPE": 20087.0, "FILE_DATA": {"edId": None,
                                                              "edNum": "5446AB28D784983CD36B75C9DF6F1925D235F9C143B08830B6CF4F3F37128ACF",
                                                              "fname": "DOC_1000859545567.pdf", "ficon": "ft/file_PDF.gif",
                                                              "attExist": True},
                        "PLAN_ENTITY_DOC_NUM": "33ABBC66CF54FDCACD65378F84E91C9E77C8081B143B9CB8FAA5A7D301463B3E"}
        program_number = "101-1048420"
    #
# print(number)
# # Print the JSON data
#
#
#
# # Load the data into a Python object
# parsed_data = json.loads(data)
#
# # Extract the URL
# url = parsed_data["features"][0]["attributes"]["pl_url"]
#
# # Extract the number from the URL
# number = url.split('/')[-2]
#
# print(number)
