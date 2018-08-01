"""Utilities that are needed for the experiments"""

import httplib2
from string import ascii_uppercase

from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials


def get_credentials(scopes, credential_path=".credentials"):
    """Create service account credentials read from a file."""
    return ServiceAccountCredentials.from_json_keyfile_name(
        credential_path,
        scopes
    )


def get_spreadsheets(credential_path=".credentials"):
    """Build the spreadsheets api given a credentials file."""
    credentials = get_credentials(
        ["https://www.googleapis.com/auth/spreadsheets"],
        credential_path
    )
    assert credentials, "No credentials found"
    assert not credentials.invalid, "The credentials are invalid"

    http = credentials.authorize(httplib2.Http())
    service = discovery.build(
        "sheets",
        "v4",
        http=http,
        discoveryServiceUrl=("https://sheets.googleapis.com/$discovery/rest?"
                             "version=v4")
    )

    return service

def append_to_spreadsheet(spreadsheet_id, sheet, data,
                          credential_path=".credentials"):
    """Append data to a spreadsheet"""
    if len(data) == 0:
        return

    sheet_range = "%s!A1:%s1" % (
        sheet,
        max(map(len, data))
    )

    sheets = get_spreadsheets(credential_path)
    return sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=sheet_range,
        body={
            "values": data,
            "majorDimension": "ROWS"
        },
        valueInputOption="USER_ENTERED"
    ).execute()
