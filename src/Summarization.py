from langchain import PromptTemplate
from langchain.schema import  HumanMessage, SystemMessage

class Summarization:
    def __init__(self) -> None:
        template= """
            Provide a summary of the following text in structured bullet point lists.
            the text: {text}
            In addition, add the provided title at the top of your response:
            the title: {title}
            """
        self.prompt=PromptTemplate(
                input_variables=["text",'title'],
                template=template
                )     
        self.formattedPrompt=''
    
    def setFormattedPrompt(self,text,title):
        self.formattedPrompt= self.prompt.format(text=text,title=title)
        
    def getFormattedPrompt(self):
        return self.formattedPrompt
    
    def getSummary(self,modelName,llm):
        summary=''
        if modelName=='ChatOpenAI':
            summary=llm.predict_messages([HumanMessage(content= self.formattedPrompt)]).content
           
        else:
            summary=llm(self.formattedPrompt)
        return summary
        