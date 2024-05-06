from pydantic import BaseModel


class PromptTemplate(BaseModel):
    template: str

    @staticmethod
    def from_template(template_str: str) -> "PromptTemplate":
        return PromptTemplate(template=template_str)

    def format(self, **args):
        return self.template.format(**args)
