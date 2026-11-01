
import pandas as pd
import great_expectations as gx


def validate_dataset(df):

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas(
        name="datasentinel_datasource"
    )

    asset = datasource.add_dataframe_asset(
        name="uploaded_dataset"
    )

    batch_definition = asset.add_batch_definition_whole_dataframe(
        "uploaded_dataset_batch"
    )

    output = []

    def run_check(check_name, column, expectation):

        suite = gx.ExpectationSuite(
            name=f"suite_{len(output)}"
        )

        suite.add_expectation(expectation)
        context.suites.add(suite)

        validation = gx.ValidationDefinition(
            name=f"validation_{len(output)}",
            data=batch_definition,
            suite=suite
        )

        context.validation_definitions.add(validation)

        result_container = validation.run(
            batch_parameters={"dataframe": df}
        )

        # Only one expectation is executed in this run
        result = result_container.results[0]
        details = result.result or {}

        output.append({
            "Validation Check": check_name,
            "Column": column,
            "Status": "PASS" if result.success else "FAIL",
            "Unexpected Count": details.get(
                "unexpected_count", 0
            ),
            "Unexpected Examples": str(
                details.get("partial_unexpected_list", [])
            )
        })

    # 1. Missing-value checks
    for column in df.columns:
        run_check(
            "No Missing Values",
            column,
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column
            )
        )

    # 2. Age validation
    for column in df.columns:
        if "age" in str(column).lower():
            run_check(
                "Age must be between 1 and 100",
                column,
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=1,
                    max_value=100
                )
            )

    # 3. Marks validation
    for column in df.columns:
        if any(word in str(column).lower()
               for word in ["mark", "score"]):
            run_check(
                "Marks must be between 0 and 100",
                column,
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=0,
                    max_value=100
                )
            )

    # 4. Attendance validation
    for column in df.columns:
        if "attendance" in str(column).lower():
            run_check(
                "Attendance must be between 0 and 100",
                column,
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=0,
                    max_value=100
                )
            )

    # 5. Student ID uniqueness
    for column in df.columns:
        if (
            "student_id" in str(column).lower()
            or str(column).lower() == "id"
        ):
            run_check(
                "Student ID must be unique",
                column,
                gx.expectations.ExpectColumnValuesToBeUnique(
                    column=column
                )
            )

    return pd.DataFrame(output)
