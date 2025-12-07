
# Schemas for data extraction

GENERIC_TEMPLATE = {
    "description": "Standard schema for generic document extraction. The LLM will try to fit data into this structure.",
    "tables": [
        {
            "table_name": "Table Name or Description",
            "sheet_or_page": "Sheet1 or Page 1",
            "columns": ["Column1", "Column2"],
            "rows": [["row1_col1", "row1_col2"]]
        }
    ]
}

# Detailed schema matching the user's input_files/output_json_format.ipynb structure
PROFIT_LOSS_TEMPLATE = {
    "description": "Comprehensive financial data extraction schema. Extract all available data for Profit & Loss, Balance Sheet, Cash Flow, and share price history.",
    "metadata": {
        "generated_date": "YYYY-MM-DD",
        "source_files": [],
        "total_companies": 0,
        "data_period": {
            "annual": "start_date to end_date",
            "quarterly": "start_date to end_date"
        }
    },
    "companies": [
        {
            "company_id": "TICKER_OR_NAME",
            "company_name": "FULL COMPANY NAME",
            "metadata": {
                "face_value": 0.0,
                "current_price": 0.0,
                "market_capitalization_cr": 0.0,
                "number_of_equity_shares": 0,
                "adjusted_equity_shares_cr": 0.0,
                "latest_version": "2.1",
                "current_version": "2.1"
            },
            "profit_loss": {
                "annual": [
                    {
                        "date": "YYYY-MM-DD",
                        "year": 2024,
                        "sales_cr": 0.0,
                        "raw_material_cost_cr": 0.0,
                        "change_in_inventory_cr": 0.0,
                        "power_and_fuel_cr": 0.0,
                        "other_manufacturing_expenses_cr": 0.0,
                        "employee_cost_cr": 0.0,
                        "selling_and_admin_expenses_cr": 0.0,
                        "other_expenses_cr": 0.0,
                        "other_income_cr": 0.0,
                        "depreciation_cr": 0.0,
                        "interest_cr": 0.0,
                        "profit_before_tax_cr": 0.0,
                        "tax_cr": 0.0,
                        "net_profit_cr": 0.0,
                        "dividend_amount_cr": 0.0,
                        "operating_profit_cr": 0.0,
                        "opm_percent": 0.0
                    }
                ],
                "quarters": {
                    "quarterly": [
                        {
                            "date": "YYYY-MM-DD",
                            "year": 2024,
                            "quarter": "Q1",
                            "sales_cr": 0.0,
                            "expenses_cr": 0.0,
                            "other_income_cr": 0.0,
                            "depreciation_cr": 0.0,
                            "interest_cr": 0.0,
                            "profit_before_tax_cr": 0.0,
                            "tax_cr": 0.0,
                            "net_profit_cr": 0.0,
                            "operating_profit_cr": 0.0,
                            "opm_percent": 0.0
                        }
                    ]
                }
            },
            "balance_sheet": {
                "annual": [
                    {
                        "date": "YYYY-MM-DD",
                        "year": 2024,
                        "equity_share_capital_cr": 0.0,
                        "reserves_cr": 0.0,
                        "borrowings_cr": 0.0,
                        "other_liabilities_cr": 0.0,
                        "total_liabilities_cr": 0.0,
                        "net_block_cr": 0.0,
                        "capital_work_in_progress_cr": 0.0,
                        "investments_cr": 0.0,
                        "other_assets_cr": 0.0,
                        "total_assets_cr": 0.0,
                        "receivables_cr": 0.0,
                        "inventory_cr": 0.0,
                        "cash_and_bank_cr": 0.0,
                        "face_value_per_share": 0.0
                    }
                ]
            },
            "cash_flow": {
                "annual": [
                    {
                        "date": "YYYY-MM-DD",
                        "year": 2024,
                        "cash_from_operating_activity_cr": 0.0,
                        "cash_from_investing_activity_cr": 0.0,
                        "cash_from_financing_activity_cr": 0.0,
                        "net_cash_flow_cr": 0.0
                    }
                ]
            }
        }
    ]
}

# Mapping of classified types to templates
TEMPLATE_MAP = {
    "profit_and_loss": PROFIT_LOSS_TEMPLATE,
    "income_statement": PROFIT_LOSS_TEMPLATE,
    "financial_statement": PROFIT_LOSS_TEMPLATE,
    "balance_sheet": PROFIT_LOSS_TEMPLATE, # Often appear together, better to use the master template
    "cash_flow": PROFIT_LOSS_TEMPLATE,
    "generic": GENERIC_TEMPLATE
}
