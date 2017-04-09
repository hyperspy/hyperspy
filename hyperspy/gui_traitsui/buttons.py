import traitsui.api as tu

OurOKButton = tu.Action(name="OK",
                        action="OK",)

OurApplyButton = tu.Action(name="Apply",
                           action="apply")

OurResetButton = tu.Action(name="Reset",
                           action="reset")

OurCloseButton = tu.Action(name="Close",
                           action="close_directly")

OurFindButton = tu.Action(name="Find next",
                          action="find",)

OurPreviousButton = tu.Action(name="Find previous",
                              action="back",)

OurFitButton = tu.Action(name="Fit",
                         action="fit")

StoreButton = tu.Action(name="Store",
                        action="store")

SaveButton = tu.Action(name="Save",
                       action="save")
